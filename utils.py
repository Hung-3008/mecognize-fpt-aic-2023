import re
from paddleocr import PaddleOCR,draw_ocr
import cv2
import json
import base64
import numpy as np

class Args:
    def __init__(self):
        self.alpha = 1.0
        self.benchmark = False
        self.beta = 1.0
        self.cpu_threads = 10
        self.crop_res_save_dir = './output'
        self.det_algorithm = 'DB'
        self.det_box_type = 'quad'
        self.det_db_box_thresh = 0.6
        self.det_db_score_mode = 'fast'
        self.det_db_thresh = 0.3
        self.det_db_unclip_ratio = 1.5
        self.det_model_dir = './models/text_detection'
        self.drop_score = 0.5
        self.enable_mkldnn = True
        self.fourier_degree = 5
        self.gpu_id = 0
        self.gpu_mem = 500
        self.image_dir = './images/'
        self.image_orientation = False
        self.ir_optim = True
        self.kie_algorithm = 'LayoutXLM'
        self.label_list = ['0', '180']
        self.ocr = True
        self.ocr_order_method = 'tb-yx'
        self.output = './output'
        self.page_num = 0
        self.precision = 'fp32'
        self.process_id = 0
        self.re_model_dir = None
        self.rec_algorithm = "SVTR_LCNet"
        self.rec_batch_num = 6
        self.rec_char_dict_path = "models/vie_dict.txt"
        self.rec_image_inverse = True
        self.rec_image_shape = '3, 48, 320'
        self.rec_model_dir = "./models/text_recognition"
        self.recovery = False
        self.return_word_box = False
        self.save_crop_res = False
        self.save_log_path = './log_output/'
        self.scales = [8, 16, 32]
        self.ser_dict_path = './models/class_list.txt'
        self.ser_model_dir = './models/ser'
        self.show_log = False
        self.total_process_num = 1
        self.use_angle_cls = True
        self.use_dilation = False
        self.use_gpu = False
        self.use_mp = False
        self.use_npu = False
        self.use_onnx = False
        self.use_pdf2docx_api = False
        self.use_pdserving = False
        self.use_space_char = True
        self.use_tensorrt = False
        self.use_visual_backbone = True
        self.use_xpu = False
        self.vis_font_path = './fonts/arial.ttf'
        self.warmup = False
        self.ocr_engine = PaddleOCR(
            use_angle_cls=False,
            det_model_dir="./models/text_detection",
            rec_model_dir="./models/text_recognition",
            rec_image_shape="3,48,320", 
            max_text_length=50,
            show_log=False,
            rec_char_dict_path="./models/vie_dict.txt",
            use_gpu=False,
            enable_mkldnn=True, 
            cls=False)

def map_quantity_usage_to_brandname(boxes):
    # Lọc các bounding box có nhãn là "brandname", "quantity", và "usage"
    # Tạo một dict để lưu trữ các biến dựa trên nhãn
    label_boxes = {}

    # Vòng for duyệt qua tất cả các box và sắp xếp chúng vào các biến dựa trên nhãn
    for box in boxes:
        label = box['pred']
        if label not in label_boxes:
            label_boxes[label] = []
        label_boxes[label].append(box)

    # Sau khi vòng for hoàn thành, bạn có thể truy cập các biến dựa trên nhãn như sau:
    brandnames = label_boxes.get('BRANDNAME', [])
    quantities = label_boxes.get('QUANTITY', [])
    usages = label_boxes.get('USAGE', [])
    diagnoses = label_boxes.get('DIAGNOSE', [])
    diagnoses = sorted(diagnoses, key=lambda box: box['points'][0][1])
    diagnoses = [box['transcription'] + ' ' for box in diagnoses]
    
    dates = label_boxes.get('DATE', [])
    dates = sorted(dates, key=lambda box: box['points'][0][1])
    dates = [box['transcription'] + ' ' for box in dates]
    
    

    # Tạo một danh sách để lưu trữ ánh xạ
    mappings = []

    for brandname in brandnames:
        # Tìm "quantity" gần nhất cho "brandname"
        nearest_quantity = None
        min_distance = float('inf')
        
        for quantity in quantities:
            # Tính khoảng cách dọc giữa "quantity" và "brandname"
            distance = abs(brandname['points'][0][1] - quantity['points'][0][1])
            
            if distance < min_distance:
                min_distance = distance
                nearest_quantity = quantity
        
        # Tìm "usage" gần nhất cho "brandname"
        nearest_usage = None
        min_distance = float('inf')
        
        for usage in usages:
            # Tính khoảng cách dọc giữa "usage" và "brandname"
            distance = abs(brandname['points'][0][1] - usage['points'][0][1])
            
            if distance < min_distance:
                min_distance = distance
                nearest_usage = usage

        # Nếu tìm thấy "quantity" gần nhất, ánh xạ "quantity" vào "brandname"
        if nearest_quantity:
            mappings.append({
                'brandname': brandname['transcription'],
                'quantity': nearest_quantity['transcription'],
                'usage': None  # Khởi tạo usage là None
            })

            # Sau khi ánh xạ, loại bỏ "quantity" khỏi danh sách "quantities" để tránh ánh xạ lại
            quantities.remove(nearest_quantity)

        # Nếu tìm thấy "usage" gần nhất, ánh xạ "usage" vào "brandname"
        if nearest_usage:
            # Tìm ánh xạ tương ứng của "quantity" (nếu có)
            corresponding_mapping = next((mapping for mapping in mappings if mapping['brandname'] == brandname['transcription']), None)
            
            # Nếu tìm thấy ánh xạ của "quantity", thêm "usage" vào nó
            if corresponding_mapping:
                corresponding_mapping['usage'] = nearest_usage['transcription']

    return mappings, diagnoses, dates

def correct_text(model_predictor, unacc_paragraphs):
    outs = ''
    for i, p in enumerate(unacc_paragraphs):
        outs += model_predictor.predict(p.strip(), NGRAM=6) + ' '
    return outs

def nomalize_bradname(text):
    pattern = r"^[\d)+*-\. ]+(.*)"
    text = re.sub(pattern, r"\1", text, flags=re.M)
    pattern = r"^[\d/#*@\-_.]+\s*(.*)$"
    cleaned = re.sub(pattern, r"\1", text, flags=re.M)
    return cleaned

def nomalize_usage(s):
    REMOVE_LIST = ['cách dùng', 'Cách dùng', 'cách', 'Cách', 'cách dùng:', 'Cách dùng:', 
                   'ghi chú', 'Ghi chú', 'Uống:', 'uống:', 'Uống', 'uống']
    remove = '|'.join(REMOVE_LIST)
    regex = re.compile(r'\b('+remove+r')\b', flags=re.IGNORECASE)
    out = regex.sub("", s)
    output_string = re.sub(r'^[^a-zA-Z0-9]+', '', out)
    return output_string.rstrip()


def nomalize_quantity(s):
    s = s.lower()
    REMOVE_LIST = ['Số lượng: ','SL:', 'sl:', 'SL', 'sl', 'Số lượng:', 'số lượng:', 'Liều lượng:', 
                   'liều lượng:', 'Liều lượng', 'liều lượng']
    remove = '|'.join(REMOVE_LIST)
    regex = re.compile(r'\b('+remove+r')\b', flags=re.IGNORECASE)
    out = regex.sub("", s)
    output_string = re.sub(r'^[^a-zA-Z0-9]+', '', out)
    return output_string.rstrip()


def nomalize_diagnose(s):
    REMOVE_LIST = ['chẩn đoán', 'Chẩn đoán', 'chẩn đoán ', 'Chẩn đoán ', 'CHẨN ĐOÁN', 'CHẨN ĐOÁN ',
                   'chẩn đoán:', 'Chẩn đoán:', 'chẩn đoán: ', 'Chẩn đoán: ', 'CHẨN ĐOÁN:', 'CHẨN ĐOÁN: ',
                   'chuẩn đoán:', 'Chuẩn đoán:', 'chuẩn đoán: ', 'Chuẩn đoán: ', 'CHUẨN ĐOÁN:', 'CHUẨN ĐOÁN: ',
                   'chuẩn đoán', 'Chuẩn đoán', 'chuẩn đoán ', 'Chuẩn đoán ', 'CHUẨN ĐOÁN', 'CHUẨN ĐOÁN ']
    remove = '|'.join(REMOVE_LIST)
    regex = re.compile(r'\b('+remove+r')\b', flags=re.IGNORECASE)
    out = regex.sub("", s)
    output_string = re.sub(r'^[^a-zA-Z0-9]+', '', out)
    return output_string.rstrip()

def text_to_json(diagnoes, medicines_list, date):
    response = {
        'date': '',
        'medicines':[],
        'diagnose':'',
        'status': 200
    }
    brand_list = []
    for medicine in medicines_list:
        temp_medicine = {
          'brandname': '',
          'usage': '',
          'quantity': ''
        }

        if len(medicine['brandname'][0]) > 0:
            brand_list.append(medicine['brandname'][0])
            temp_medicine['brandname'] = nomalize_bradname(medicine['brandname'][0])
            for i, usage in enumerate(medicine['usage']):
                temp_medicine['usage'] += nomalize_usage(usage) + ' '
            for i, quantity in enumerate(medicine['quantity']):
                temp_medicine['quantity'] += nomalize_quantity(quantity) + ' '
            response['medicines'].append(temp_medicine)
    if len(diagnoes) > 0:
        for i, diagnose in enumerate(diagnoes):
            response['diagnose'] += nomalize_diagnose(diagnose) + ' '
    if len(date) > 0:
        for i, d in enumerate(date):
            response['date'] += d + ' '

    if len(brand_list) == 0:
        response['status'] = 702
        return response

    return response



def read_image_from_json(data):
    image_list = []
    for base64_string in data['image']:
        base64_bytes = base64_string.encode('utf-8')
        image_bytes = base64.b64decode(base64_bytes) 
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        image_list.append(image)
    return image_list



def extract_phrases(text):
    return re.findall(r'\w[\w ]+', text)



def get_brand_and_related(data):

  label_list = []
  diagnoes = []
  date = []
  for item in data: 
    try:
      if item['pred'] != 'O':
        if item['pred'] == 'DIAGNOSE':
          diagnoes.append(item['transcription'])
        elif item['pred'] == 'DATE':
          date.append(item['transcription'])
        else:
          label_list.append({'label': item['pred'], 'text': item['transcription']})
    except:
      continue

  medicines_list = []
  

  while len(label_list) > 0:
    item = label_list.pop(0)
    if item['label'] == 'BRANDNAME':
        medicines = {
            'brandname': [],
            'usage': [],
            'quantity': [],
        }
        medicines['brandname'].append(item['text'])
        while len(label_list) > 0:
            item = label_list.pop(0)
            if item['label'] == 'USAGE':
                medicines['usage'].append(item['text'])
            elif item['label'] == 'QUANTITY':
                medicines['quantity'].append(item['text'])
            else:
                label_list.insert(0, item)
                break
        medicines_list.append(medicines)
    else:
        continue
  return diagnoes, medicines_list, date
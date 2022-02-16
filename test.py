import re
from collections import deque
from datagene.tokenizers import KoBertTokenizer

tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")

START_OF_PREDICATE_SPECIAL_TOKEN = "<PREDICATE>"
END_OF_PREDICATE_SPECIAL_TOKEN = "</PREDICATE>"
    
tokenizer.add_special_tokens({"additional_special_tokens":[
    START_OF_PREDICATE_SPECIAL_TOKEN,
    END_OF_PREDICATE_SPECIAL_TOKEN
]})
        
# predicate_begin_idx, predicate_end_idx = 75, 79
# sentence =  "코오롱글로벌은 강원 춘천시에 입지를 정했고, 여수 경도레저관광과 중국 평강도가촌 등을 포함한 컨소시엄 3곳은 전남 여수 경도를 입지로 택했다."
# arguments =  [
#     {'form': '여수 경도레저관광과 중국 평강도가촌 등을 포함한 컨소시엄 3곳', 'label': 'ARG0', 'begin': 25, 'end': 59, 'word_id': 13}, 
#     {'form': '전남 여수 경도', 'label': 'ARG1', 'begin': 61, 'end': 69, 'word_id': 16}, 
#     {'form': '입지', 'label': 'ARG3', 'begin': 71, 'end': 73, 'word_id': 17}
# ]

predicate_begin_idx, predicate_end_idx = 0, 3
sentence =  "썰렁한 코미디 장면이 많으니 주의할 것."
arguments =  [
    {'form': '썰렁한 코미디 장면', 'label': 'ARG0', 'begin': 0, 'end': 10, 'word_id': 1}, 
]


# predicate_begin_idx, predicate_end_idx = 72, 77
# sentence =  "참석자들은 특히 지난해 사업권 심사에서 탈락한 롯데면세점 월드타워점과 SK네트웍스의 워커힐면세점이 다시 특허를 받는 상황에 대해 우려했다."
# arguments =  [
#     {'form': '참석자들', 'label': 'ARG0', 'begin': 0, 'end': 4, 'word_id': 1}, 
#     {'form': '대해', 'label': 'ARG1', 'begin': 69, 'end': 71, 'word_id': 15}
# ]

# predicate_begin_idx, predicate_end_idx = 25, 27
# sentence =  "20여년 뒤 한국에서 군 정찰기 도입 로비를 하다 현직 국방장관과 '연서(戀書) 스캔들'을 일으킨 재미 무기중개상 린다 김이었다."
# arguments =  [
#     {'form': '20여년 뒤', 'label': 'ARGM-TMP', 'begin': 0, 'end': 6, 'word_id': 2}, 
#     {'form': '한국', 'label': 'ARGM-LOC', 'begin': 7, 'end': 9, 'word_id': 3}, 
#     {'form': '군 정찰기 도입 로비', 'label': 'ARG1', 'begin': 12, 'end': 23, 'word_id': 7}]

original_sentence = sentence
sentence = sentence[:predicate_begin_idx] + START_OF_PREDICATE_SPECIAL_TOKEN + sentence[predicate_begin_idx: predicate_end_idx] + END_OF_PREDICATE_SPECIAL_TOKEN + sentence[predicate_end_idx:]

token_list = tokenizer.tokenize(sentence)
label_list = []

for idx in range(len(arguments)):
    arg_end_idx = int(arguments[idx]["end"])
    
    if arg_end_idx > predicate_end_idx:
        arguments[idx]["begin"] += 2#(len(START_OF_PREDICATE_SPECIAL_TOKEN) + len(END_OF_PREDICATE_SPECIAL_TOKEN))
        arguments[idx]["end"] += 2#(len(START_OF_PREDICATE_SPECIAL_TOKEN) + len(END_OF_PREDICATE_SPECIAL_TOKEN))

# Convert word positions to char positions
char_label_list = ["O" for _ in range(len(original_sentence) + 2)]

for argument in arguments:
    begin_idx = int(argument["begin"])
    end_idx = int(argument["end"])
    
    char_label_list[begin_idx: end_idx] = [argument["label"]] * (end_idx - begin_idx)

# Convert char positions to token positions
char_label_list = deque(char_label_list)

for token_idx, token in enumerate(token_list):
    label = "O"
    
    if token_idx == 0 and token.startswith("▁"):
        token = token[1:]
    
    if token == START_OF_PREDICATE_SPECIAL_TOKEN or token == END_OF_PREDICATE_SPECIAL_TOKEN:
        char_label_list.popleft()
        label_list.append(label)
        continue
    
    while token:
        char_label = char_label_list.popleft()
        
        token = token[1:]
        
        if char_label != "O":
            label = char_label
    
    label_list.append(label)
    
# BIOES Tagging
previous_label = "O"
for idx in range(len(label_list)):
    label = label_list[idx]
    
    if label == "O":
        previous_label = "O"
        continue
    
    if previous_label != label:
        if idx != len(label_list) - 1:
            if label == label_list[idx + 1]:
                label_list[idx] = "B-" + label
            else:
                label_list[idx] = "S-" + label
        else:
            label_list[idx] = "S-" + label
            
    else:
        if idx == len(label_list) - 1:
            label_list[idx] = "E-" + label     
        else:
            if label == label_list[idx + 1]:
                label_list[idx] = "I-" + label
            else:
                label_list[idx] = "E-" + label
    
    previous_label = label
        
        
            # if previous_label == "O":
            #     label_list[idx] = "S-" + label_list[idx]
        
        
        
        
    




# for idx in range(len(arguments)):
#     arg_end_idx = int(arguments[idx]["end"])
    
#     if arg_end_idx > predicate_end_idx:
#         arguments[idx]["begin"] += (len(START_OF_PREDICATE_SPECIAL_TOKEN) + len(END_OF_PREDICATE_SPECIAL_TOKEN))
#         arguments[idx]["end"] += (len(START_OF_PREDICATE_SPECIAL_TOKEN) + len(END_OF_PREDICATE_SPECIAL_TOKEN))
        
# sentence = sentence[:predicate_begin_idx] + START_OF_PREDICATE_SPECIAL_TOKEN + sentence[predicate_begin_idx: predicate_end_idx] + END_OF_PREDICATE_SPECIAL_TOKEN + sentence[predicate_end_idx:]

# trigger = True

# token_list = tokenizer.tokenize(sentence)
# label_list = ["O"] * len(token_list)

# previous_token_length = 0

# for token_idx, token in enumerate(token_list):    
#     if token.startswith("▁"):
#         t = token[1:]
#         if token_idx == 0:
#             token_start_idx = previous_token_length
#         else:
#             token_start_idx = previous_token_length + 1
#     else:
#         t = token
#         token_start_idx = previous_token_length
    
#     token_end_idx = token_start_idx + len(t) - 1
    
#     previous_token_length = token_end_idx + 1
    
#     if not arguments:
#         break
    
#     argument = arguments[0]
#     arg_start_idx, arg_end_idx = int(argument["begin"]), int(argument["end"]) - 1
#     arg_label = argument["label"]
    
#     if token_start_idx <= arg_start_idx <= token_end_idx or token_start_idx <= arg_end_idx <= token_end_idx:
#         label_list[token_idx] = f"{arg_label}"
        
#         if arg_end_idx == token_end_idx:
#             del arguments[0]
        
#     elif arg_start_idx <= token_start_idx <= arg_end_idx or arg_start_idx <= token_end_idx <= arg_end_idx:
#         label_list[token_idx] = f"{arg_label}"
       
        
for token, label in zip(token_list, label_list):
    print(token, label)
    
    
    
"""
predicate :  75 79
Sentence :  코오롱글로벌은 강원 춘천시에 입지를 정했고, 여수 경도레저관광과 중국 평강도가촌 등을 포함한 컨소시엄 3곳은 전남 여수 경도를 입지로 <PREDICATE>택했다.</PREDICATE>
Arguments :  [{'form': '여수 경도레저관광과 중국 평강도가촌 등을 포함한 컨소시엄 3곳', 'label': 'ARG0', 'begin': 25, 'end': 59, 'word_id': 13}, {'form': '전남 여수 경도', 'label': 'ARG1', 'begin': 61, 'end': 69, 'word_id': 16}, {'form': '입지', 'label': 'ARG3', 'begin': 71, 'end': 73, 'word_id': 17}]



predicate :  72 77
Sentence :  참석자들은 특히 지난해 사업권 심사에서 탈락한 롯데면세점 월드타워점과 SK네트웍스의 
워커힐면세점이 다시 특허를 받는 상황에 대해 <PREDICATE>우려했다.</PREDICATE>
Arguments :  [{'form': '참석자들', 'label': 'ARG0', 'begin': 0, 'end': 4, 'word_id': 1}, {'form': '대해', 'label': 'ARG1', 'begin': 69, 'end': 71, 'word_id': 15}]


predicate :  25 27
Sentence :  20여년 뒤 한국에서 군 정찰기 도입 로비를 <PREDICATE>하다</PREDICATE> 현직 국방장관 
과 '연서(戀書) 스캔들'을 일으킨 재미 무기중개상 린다 김이었다.
Arguments :  [{'form': '20여년 뒤', 'label': 'ARGM-TMP', 'begin': 0, 'end': 6, 'word_id': 2}, {'form': '한국', 'label': 'ARGM-LOC', 'begin': 7, 'end': 9, 'word_id': 3}, {'form': '군 정찰기  
도입 로비', 'label': 'ARG1', 'begin': 12, 'end': 23, 'word_id': 7}]
"""
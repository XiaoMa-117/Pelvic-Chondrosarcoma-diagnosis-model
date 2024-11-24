import joblib
import numpy as np
import pandas as pd

# 从文件中加载模型和 scaler
loaded_model = joblib.load('random_forest_model.pkl')
# scaler = joblib.load('scaler.pkl')
# 输入数据
# example_input = [1, 43, 2, 1, 0, 1, 1]
# 输入
example_input = [0,0,0,0,0,0,0]
example_input[0] = input("What is the patient's gender? Please input '0' for female or '1' for male.\n")
example_input[1] = input("What is the patient's age? Please input only one number.\n")
example_input[2] = input("Where is the tumor located in the pelvis? Please input '1' for ilium '2' for acetabulum, or '3' for ischium and pubis.\n")
example_input[3] = input("Does the patient has an soft tissue mass as determined by radiological examinations? Please input '0' for no or '1' for yes.\n")
example_input[4] = input("Does the tumor feature high signal intensity on T2-weighted MRI? Please input '0' for no or '1' for yes.\n")
example_input[5] = input("Does the tumor feature a ring-and-arc enhancement pattern on contrast-enhanced T1-weighted MRI? Please input '0' for no or '1' for yes.\n")
example_input[6] = input("Does the tumor feature intratumoral calcification? Please input '0' for no or '1' for yes.\n")
# 将输入转换为二维数组
example_input_array = np.array(example_input).reshape(1, -1)
# 使用加载的 scaler 对输入数据进行标准化处理
# example_input_scaled = scaler.transform(example_input_array)
# 使用加载的模型进行预测
predicted_class = loaded_model.predict(pd.DataFrame(example_input_array))

if predicted_class[0] == 0:
    finalresult = "The patient is predicted to have pelvic chondrosarcoma."
else:
    finalresult = "The patient is predicted to not have pelvic chondrosarcoma."

print(finalresult)

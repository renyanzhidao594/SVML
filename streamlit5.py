#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn.svm import SVC
import os

# 解决matplotlib绘图潜在后端问题
plt.switch_backend('Agg')

# -------------------------- 1. 路径配置（可根据实际情况修改） --------------------------
# 模型/标准化器/特征名/阈值路径（相对路径，需与代码放在同一目录）
MODEL_PATH = "svm_best_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"
THRESHOLD_PATH = "best_thresholds.pkl"
# 训练集路径（关键：需提前在训练脚本中保存训练集，这里填写训练集pkl文件路径）
TRAIN_DATA_PATH = "train_data.pkl"  # 训练集保存为pkl格式（X_train_df）

# -------------------------- 2. 加载模型、标准化器、特征名、最佳阈值 --------------------------
try:
    # 加载线性SVM模型并验证
    model = joblib.load(MODEL_PATH)
    if not (isinstance(model, SVC) and model.kernel == "linear"):
        st.error("加载的模型不是线性SVM，请检查模型文件路径！")
        st.stop()

    # 加载标准化器
    scaler = joblib.load(SCALER_PATH)

    # 加载训练时的特征列顺序
    FEATURE_NAMES = joblib.load(FEATURE_NAMES_PATH)

    # 加载训练时的最佳阈值
    best_thresholds = joblib.load(THRESHOLD_PATH)
    svm_best_thresh = best_thresholds['svm']

    # 调试信息
    st.success("模型、标准化器、特征配置加载成功！")
    st.info(f"特征顺序（与训练集一致）：{FEATURE_NAMES}")
    st.info(f"SVM最佳判定阈值：{svm_best_thresh}")
    st.info(f"模型类别映射：{model.classes_}")
    st.info(f"Scaler前5个特征均值：{scaler.mean_[:5]}")

except Exception as e:
    st.error(f"加载失败：{str(e)}，请检查文件路径是否正确！")
    st.stop()

# -------------------------- 3. 网页输入界面 --------------------------
st.title("Gastric Cancer Liver Metastasis Predictor")
st.subheader("Input Feature Values")

user_input_dict = {}

# 按训练集特征顺序生成输入框
for feat in FEATURE_NAMES:
    if feat == "hb":
        user_input_dict[feat] = st.number_input("Hemoglobin (Hb, g/L)", min_value=30.0, max_value=200.0, value=120.0, step=0.1)
    elif feat == "tt":
        user_input_dict[feat] = st.number_input("Thrombin Time (TT, sec)", min_value=5.0, max_value=30.0, value=13.0, step=0.1)
    elif feat == "siri":
        user_input_dict[feat] = st.number_input("Systemic Inflammatory Response Index (SIRI)", min_value=0.0, max_value=10.0, value=1.0, step=0.01)
    elif feat == "afr":
        user_input_dict[feat] = st.number_input("Albumin to Fibrinogen Ratio (AFR)", min_value=0.1, max_value=20.0, value=2.5, step=0.01)
    elif feat == "cea_log":
        # CEA原始值自动转换为cea_log
        cea_original = st.number_input("CEA Original Value (ng/mL)", min_value=0.0, max_value=1000.0, value=1.47, step=0.1)
        cea_log = np.log10(cea_original + 1)
        user_input_dict[feat] = cea_log
        st.caption(f"CEA原始值={cea_original} → cea_log≈{cea_log:.6f}")
    elif feat == "lvi":
        user_input_dict[feat] = st.selectbox("Lymphovascular Invasion (LVI)", options=[0, 1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
    elif feat == "t_stage":
        user_input_dict[feat] = st.selectbox("T-stage", options=[1, 2, 3, 4], format_func=lambda x: f"T{x}")
    elif feat == "n_stage":
        user_input_dict[feat] = st.selectbox("N-stage", options=[0, 1, 2, 3], format_func=lambda x: f"N{x}")
    # 兼容其他未明确指定的特征（防止KeyError）
    else:
        user_input_dict[feat] = st.number_input(f"{feat} (Input Value)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

# -------------------------- 4. 构造模型输入 --------------------------
try:
    user_input_list = [user_input_dict[feat] for feat in FEATURE_NAMES]
    input_data = np.array(user_input_list).reshape(1, -1)
except KeyError as e:
    st.error(f"特征缺失：{str(e)}，请检查特征名称是否一致！")
    st.stop()

# -------------------------- 5. 预测逻辑（核心修改部分） --------------------------
if st.button("Predict Liver Metastasis Risk"):
    # 标准化输入数据
    input_data_scaled = scaler.transform(input_data)

    # 定义阳性/阴性类别
    positive_class = 1  # 有肝转移（阳性）
    negative_class = 0  # 无肝转移（阴性）

    # 模型预测概率
    try:
        # 先验证模型是否开启概率预测
        if not hasattr(model, 'predict_proba'):
            st.error("SVM模型未开启概率预测！请重新训练模型并设置 probability=True")
            st.stop()

        predicted_proba = model.predict_proba(input_data_scaled)[0]
        # 核心修改1：动态获取阳性类别在model.classes_中的索引（不再硬编码[1]）
        positive_class_index = list(model.classes_).index(positive_class)
        # 核心修改2：使用阳性类别对应索引的概率与阈值对比，判定类别
        predicted_class = positive_class if predicted_proba[positive_class_index] >= svm_best_thresh else negative_class
    except AttributeError:
        st.warning("模型未开启概率预测，无法显示概率值！请重新训练SVM并设置probability=True")
        predicted_proba = [0, 0]
        predicted_class = model.predict(input_data_scaled)[0]
        # 动态获取正负类别索引（兼容降级场景）
        positive_class_index = list(model.classes_).index(positive_class) if positive_class in model.classes_ else 0
        negative_class_index = list(model.classes_).index(negative_class) if negative_class in model.classes_ else 1

    # 展示预测结果（优化后：判定逻辑与概率提取完全一致，无脱节）
    st.write("### Prediction Result")
    # 核心修改3：动态获取无转移/有转移概率，与判定逻辑关联
    no_metastasis_prob = predicted_proba[list(model.classes_).index(negative_class)] * 100 if len(predicted_proba) > 0 else 0
    metastasis_prob = predicted_proba[list(model.classes_).index(positive_class)] * 100 if len(predicted_proba) > 0 else 0

    if predicted_class == positive_class:
        st.error(f"**Liver Metastasis Risk: High Risk**")
        st.write(f"Probability of Liver Metastasis: {metastasis_prob:.1f}%")
    else:
        st.success(f"**Liver Metastasis Risk: Low Risk**")
        st.write(f"Probability of No Liver Metastasis: {no_metastasis_prob:.1f}%")
    # 展示参考阈值
    st.caption(f"Reference Threshold: {svm_best_thresh:.2f} (Probability of Liver Metastasis ≥ this value = High Risk)")

    # 临床建议
    st.write("### Clinical Advice")
    advice_high = (
        "1. Complete enhanced abdominal CT/MRI within 1 month to confirm metastasis;\n"
        "2. Monitor serological indicators (Hb, CEA) every 2 weeks;\n"
        "3. Consult an oncologist for adjuvant therapy (targeted/chemotherapy);\n"
        "4. Maintain a high-protein diet to improve anemia."
    )
    advice_low = (
        "1. Follow up (abdominal ultrasound + serology) every 3 months;\n"
        "2. Avoid alcohol/spicy foods to reduce gastric irritation;\n"
        "3. Keep regular schedule & moderate exercise;\n"
        "4. Seek medical help for abdominal pain/jaundice/weight loss."
    )
    st.write(advice_high if predicted_class == positive_class else advice_low)

    # -------------------------- SHAP 可视化（完全修复版） --------------------------
    st.write("### Model Interpretation (SHAP Force Plot)")
    try:
        # 修复1：加载有效背景数据（训练集）
        if os.path.exists(TRAIN_DATA_PATH):
            # 加载训练集（提前保存的X_train_df）
            train_data = joblib.load(TRAIN_DATA_PATH)
            # 仅保留训练集中的对应特征，并标准化
            train_features = train_data[FEATURE_NAMES] if isinstance(train_data, pd.DataFrame) else train_data
            background_data = scaler.transform(train_features)
            # 抽样100个样本作为背景数据（减少计算量）
            background_data = shap.sample(background_data, 100, random_state=42)
        else:
            # 备用方案：无训练集时，生成随机模拟背景数据
            st.warning("未找到训练集，使用随机模拟数据作为背景（效果有限），建议提供训练集文件！")
            background_data = np.random.randn(100, len(FEATURE_NAMES))

        # 修复2：线性SVM使用专用LinearExplainer（替代KernelExplainer）
        explainer = shap.LinearExplainer(
            model=model,
            masker=background_data,
            feature_names=FEATURE_NAMES
        )
        # 计算有效SHAP值
        shap_values = explainer.shap_values(input_data_scaled)

        # 调试：展示SHAP值（此时非0）
        st.write("#### SHAP值调试（特征贡献度）")
        shap_debug_df = pd.DataFrame(
            np.hstack([shap_values, input_data_scaled]),
            columns=[f"{f}_shap" for f in FEATURE_NAMES] + FEATURE_NAMES
        )
        st.dataframe(shap_debug_df)

        # 原生HTML渲染Force Plot（正常显示特征贡献）
        shap_html = shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values,
            features=pd.DataFrame(input_data_scaled, columns=FEATURE_NAMES),
            feature_names=FEATURE_NAMES,
            out_names="Liver Metastasis Risk",
            plot_cmap="RdBu_r",
            show=False
        )
        shap_html_str = f"<head>{shap.getjs()}</head><body>{shap_html.html()}</body>"
        components.html(shap_html_str, height=200, scrolling=True)

        # 修复3：移除单样本无效的SHAP摘要图，或替换为训练集整体摘要图（可选）
        st.write("#### SHAP Feature Importance (Global)")
        # 若有训练集，绘制全局特征重要性；若无，跳过
        if os.path.exists(TRAIN_DATA_PATH):
            # 计算训练集的SHAP值（展示全局重要性）
            train_shap_values = explainer.shap_values(background_data)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                train_shap_values,
                features=pd.DataFrame(background_data, columns=FEATURE_NAMES),
                feature_names=FEATURE_NAMES,
                plot_type="dot",
                show=False
            )
            plt.tight_layout()
            st.pyplot(plt)
        else:
            st.info("无训练集，无法绘制全局SHAP摘要图，请提供训练集文件！")

    except Exception as e:
        st.error(f"SHAP绘图失败：{str(e)}")
        st.info("可能原因：1.SVM模型未开启probability=True；2.训练集文件不存在；3.SHAP版本不兼容（建议安装shap==0.42.1）")


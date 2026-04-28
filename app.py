import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.linalg import inv
from sklearn.decomposition import PCA
import io
import warnings
warnings.filterwarnings('ignore')

# ==================== 页面配置 ====================
st.set_page_config(page_title="PCA & 马氏距离分析", layout="wide")
st.title("📊 SEC-MS 相似性评估工具（Pooled Covariance）")
st.caption("上传 Excel → Sum归一化 → Pareto Scaling → PCA → 马氏距离 → 一键下载报告")

# ==================== 文件上传 ====================
uploaded_file = st.file_uploader(
    "📁 点击或拖拽上传 Excel 文件",
    type=['xlsx', 'xls']
)

if uploaded_file is None:
    st.info("👆 请先上传 Excel 文件，支持 .xlsx / .xls 格式")
    st.stop()

# ==================== 读取数据 ====================
xl = pd.ExcelFile(uploaded_file)
sheet_name = 'Data' if 'Data' in xl.sheet_names else xl.sheet_names[0]
df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
st.success(f"✅ 已读取 sheet「{sheet_name}」，共 {df.shape[0]} 行 × {df.shape[1]} 列")

cols = list(df.columns)

# ==================== 用户选择列 ====================
with st.expander("📋 列名配置", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        g_default = next((i for i, c in enumerate(cols) if 'Group' in c or 'group' in c), 0)
        group_col = st.selectbox("分组列（如 Group）", cols, index=g_default)
    with c2:
        s_default = next((i for i, c in enumerate(cols) if 'Sample' in c or 'sample' in c or 'Name' in c), min(1, len(cols)-1))
        sample_col = st.selectbox("样本名列（如 Samplecode）", cols, index=s_default)

unique_groups = df[group_col].dropna().unique().tolist()

if len(unique_groups) < 2:
    st.error("❌ 分组列中需要至少 2 个不同的组")
    st.stop()

c3, c4 = st.columns(2)
with c3:
    target_group = st.selectbox("目标组（如 300）", unique_groups)
with c4:
    ref_default_idx = min(1, len(unique_groups) - 1)
    ref_group = st.selectbox("参考组（如 1000）", unique_groups, index=ref_default_idx)

# ==================== 高级选项 ====================
with st.expander("⚙️ 高级选项（一般不需要改）"):
    c5, c6 = st.columns(2)
    with c5:
        n_components = st.number_input("PCA 主成分数", min_value=2, max_value=10, value=3, step=1)
    with c6:
        show_score_table = st.checkbox("显示 PCA 得分明细", value=True)

    threshold = st.number_input("达标阈值", value=3.3, min_value=0.1, step=0.1)

    c7, c8 = st.columns(2)
    with c7:
        x_min, x_max = st.slider("X 轴范围", -1.0, 1.0, (-0.4, 0.5), 0.1)
    with c8:
        y_min, y_max = st.slider("Y 轴范围", -1.0, 1.0, (-0.4, 0.4), 0.1)

# ==================== 开始计算 ====================
if st.button("🚀 开始计算", type="primary", use_container_width=True):
    # 识别数值列（排除分组列和样本名列）
    feature_cols = []
    for c in cols:
        if c in [group_col, sample_col]:
            continue
        try:
            df[c].astype(float)
            feature_cols.append(c)
        except (ValueError, TypeError):
            pass

    if len(feature_cols) == 0:
        st.error("❌ 没有找到数值列，请检查数据格式")
        st.stop()

    X = df[feature_cols].values.astype(float)
    groups = df[group_col].values
    samples = df[sample_col]
    n_total = len(groups)

    idx_target = np.where(groups == target_group)[0]
    idx_ref = np.where(groups == ref_group)[0]

    if len(idx_target) == 0 or len(idx_ref) == 0:
        st.error(f"❌ 目标组「{target_group}」或参考组「{ref_group}」没有匹配样本")
        st.stop()

    n_target = len(idx_target)
    n_ref = len(idx_ref)

    st.info(f"特征数: **{len(feature_cols)}** | {target_group}: **{n_target}** 个 | {ref_group}: **{n_ref}** 个")

    with st.spinner("🔄 数据预处理 & PCA & 马氏距离计算中..."):

        # ====== 1. Sum 归一化 ======
        X_sum_norm = X / X.sum(axis=1, keepdims=True)

        # ====== 2. Pareto Scaling ======
        std_vals = np.std(X_sum_norm, axis=0)
        X_pareto = (X_sum_norm - np.mean(X_sum_norm, axis=0)) / np.sqrt(std_vals + 1e-10)

        # ====== 3. PCA ======
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(X_pareto)
        cumvar = np.cumsum(pca.explained_variance_ratio_)

        # 构建 scores DataFrame
        pc_names = [f'PC{i+1}' for i in range(n_components)]
        scores_df = pd.DataFrame(scores, columns=pc_names)
        scores_df['Group'] = groups
        scores_df['Samplecode'] = samples.values

        # ====== 4. 按分组提取得分 ======
        scores_A = scores_df[scores_df['Group'] == target_group][pc_names].values
        scores_B = scores_df[scores_df['Group'] == ref_group][pc_names].values

        # ====== 5. 马氏距离（Pooled Covariance）- 修正版 ======
        mean_A = np.mean(scores_A, axis=0)
        mean_B = np.mean(scores_B, axis=0)
        cov_A = np.cov(scores_A.T)
        cov_B = np.cov(scores_B.T)

        n_A = len(scores_A)
        n_B = len(scores_B)
        # 使用自由度 (n-1) 加权计算合并协方差矩阵
        S_pooled = ((n_A - 1) * cov_A + (n_B - 1) * cov_B) / (n_A + n_B - 2)

        diff = mean_A - mean_B
        S_inv = inv(S_pooled)
        D_M = np.sqrt(np.dot(np.dot(diff, S_inv), diff.T))

        # ====== 6. 各样本到参考组均值的个体马氏距离 ======
        individual_dists = []
        for i in idx_target:
            d_vec = scores[i] - mean_B
            d = np.sqrt(np.dot(np.dot(d_vec, S_inv), d_vec.T))
            individual_dists.append(d)

        # ====== 7. 绘制 PCA 得分图 ======
        def plot_confidence_ellipse(ax, x, y, n_std=2.15, **kwargs):
            """绘制置信椭圆（基于特征值分解）"""
            if len(x) < 3:
                return
            cov_mat = np.cov(x, y)
            lambda_vals, vecs = np.linalg.eigh(cov_mat)
            order = lambda_vals.argsort()[::-1]
            lambda_vals = lambda_vals[order]
            vecs = vecs[:, order]
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            width, height = 2 * n_std * np.sqrt(lambda_vals)
            ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                              width=width, height=height,
                              angle=angle, **kwargs)
            ax.add_patch(ellipse)

        colors_map = {target_group: '#1f77b4', ref_group: '#d62728'}
        markers_map = {target_group: 'o', ref_group: 's'}
        short_names = [str(s).split()[-1] for s in samples.values]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # ---- 左图：PC1 vs PC2 ----
        for grp in [target_group, ref_group]:
            subset = scores_df[scores_df['Group'] == grp]
            ax1.scatter(subset['PC1'], subset['PC2'],
                        label=f"{grp} (n={len(subset)})",
                        c=colors_map.get(grp, 'gray'), marker=markers_map.get(grp, 'o'),
                        s=80, alpha=0.7, edgecolors='black', linewidths=0.5)
        for grp in [target_group, ref_group]:
            subset = scores_df[scores_df['Group'] == grp]
            if len(subset) > 2:
                plot_confidence_ellipse(ax1, subset['PC1'].values, subset['PC2'].values,
                                        edgecolor=colors_map.get(grp, 'gray'),
                                        facecolor=colors_map.get(grp, 'gray'),
                                        alpha=0.08, linewidth=2, linestyle='--')
        for i in range(n_total):
            ax1.annotate(short_names[i],
                         (scores[i, 0], scores[i, 1]),
                         fontsize=5.5, xytext=(2, 2), textcoords='offset points')
        ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax1.set_title('PC1 vs PC2')
        ax1.legend(fontsize=9)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.axhline(0, color='gray', linewidth=0.5)
        ax1.axvline(0, color='gray', linewidth=0.5)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)

        # ---- 右图：PC1 vs PC3 ----
        if n_components >= 3:
            for grp in [target_group, ref_group]:
                subset = scores_df[scores_df['Group'] == grp]
                ax2.scatter(subset['PC1'], subset['PC3'],
                            label=f"{grp} (n={len(subset)})",
                            c=colors_map.get(grp, 'gray'), marker=markers_map.get(grp, 'o'),
                            s=80, alpha=0.7, edgecolors='black', linewidths=0.5)
            for grp in [target_group, ref_group]:
                subset = scores_df[scores_df['Group'] == grp]
                if len(subset) > 2:
                    plot_confidence_ellipse(ax2, subset['PC1'].values, subset['PC3'].values,
                                            edgecolor=colors_map.get(grp, 'gray'),
                                            facecolor=colors_map.get(grp, 'gray'),
                                            alpha=0.08, linewidth=2, linestyle='--')
            for i in range(n_total):
                ax2.annotate(short_names[i],
                             (scores[i, 0], scores[i, 2]),
                             fontsize=5.5, xytext=(2, 2), textcoords='offset points')
            ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax2.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
            ax2.set_title('PC1 vs PC3')
        else:
            for grp in [target_group, ref_group]:
                subset = scores_df[scores_df['Group'] == grp]
                ax2.scatter(subset['PC1'], subset['PC2'],
                            label=f"{grp} (n={len(subset)})",
                            c=colors_map.get(grp, 'gray'), marker=markers_map.get(grp, 'o'),
                            s=80, alpha=0.7, edgecolors='black', linewidths=0.5)
            ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            ax2.set_title('PC1 vs PC2 (副本)')

        ax2.legend(fontsize=9)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.axhline(0, color='gray', linewidth=0.5)
        ax2.axvline(0, color='gray', linewidth=0.5)
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)

        plt.suptitle(f"PCA 得分图 (马氏距离 D_M = {D_M:.4f})", fontsize=14)
        plt.tight_layout()

    # ==================== 展示结果 ====================
    st.divider()
    st.subheader("📈 计算摘要")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("特征数", len(feature_cols))
    m2.metric(f"前{n_components}PC 累计方差", f"{cumvar[n_components-1]*100:.1f}%")
    m3.metric("目标组样本数", n_target)
    m4.metric("参考组样本数", n_ref)
    m5.metric(f"马氏距离 D_M", f"{D_M:.4f}",
              delta="✓达标" if D_M < threshold else "✗超标",
              delta_color="normal" if D_M < threshold else "inverse")

    # 各主成分方差贡献
    pc_info_cols = st.columns(n_components + 1)
    for i in range(n_components):
        pc_info_cols[i].metric(f"PC{i+1} 方差", f"{pca.explained_variance_ratio_[i]*100:.2f}%")

    st.subheader("🗺️ PCA 得分图")
    st.pyplot(fig)
    plt.close(fig)

    # ====== 个体马氏距离结果 ======
    result_df = pd.DataFrame({
        '样本名': [samples.iloc[i] for i in idx_target],
        '短名': [short_names[i] for i in idx_target],
        '组别': target_group,
        '马氏距离(到参考组均值)': [round(d, 4) for d in individual_dists],
    })
    result_df['是否达标'] = result_df['马氏距离(到参考组均值)'].apply(
        lambda x: f"✅ 达标" if x < threshold else f"❌ 超标"
    )

    st.subheader("📏 各样本马氏距离结果")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    # PCA 得分明细
    if show_score_table:
        score_display = scores_df[['Samplecode', 'Group'] + pc_names].copy()
        for i in range(n_components):
            score_display[pc_names[i]] = score_display[pc_names[i]].round(5)
        with st.expander("📋 PCA 得分明细"):
            st.dataframe(score_display, use_container_width=True, hide_index=True)

    # ==================== 下载 ====================
    st.divider()
    st.subheader("📥 下载结果")
    d_c1, d_c2, d_c3 = st.columns(3)

    with d_c1:
        csv = result_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 马氏距离 CSV",
                           csv, "mahalanobis_results.csv", "text/csv")

    with d_c2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            summary_data = {
                '项目': ['目标组', '参考组', '目标样本数', '参考样本数',
                         '特征数', 'PCA主成分数',
                         f'前{n_components}PC累计方差',
                         '组间马氏距离 D_M', '达标阈值'],
                '值': [target_group, ref_group, n_target, n_ref,
                       len(feature_cols), n_components,
                       f"{cumvar[n_components-1]*100:.2f}%",
                       round(D_M, 4), threshold]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='汇总', index=False)
            result_df.to_excel(writer, sheet_name='马氏距离', index=False)
            score_export = scores_df[['Samplecode', 'Group'] + pc_names].copy()
            for i in range(n_components):
                score_export[pc_names[i]] = score_export[pc_names[i]].round(5)
            score_export.to_excel(writer, sheet_name='PCA得分', index=False)
        buf.seek(0)
        st.download_button("📥 完整报告 Excel",
                           buf, "PCA_mahalanobis_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with d_c3:
        img_buf = io.BytesIO()
        fig2 = plt.figure(figsize=(14, 6))
        ax_l, ax_r = fig2.subplots(1, 2)
        # 重绘左图
        for grp in [target_group, ref_group]:
            subset = scores_df[scores_df['Group'] == grp]
            ax_l.scatter(subset['PC1'], subset['PC2'],
                         label=f"{grp} (n={len(subset)})",
                         c=colors_map.get(grp, 'gray'), marker=markers_map.get(grp, 'o'),
                         s=80, alpha=0.7, edgecolors='black', linewidths=0.5)
        for grp in [target_group, ref_group]:
            subset = scores_df[scores_df['Group'] == grp]
            if len(subset) > 2:
                plot_confidence_ellipse(ax_l, subset['PC1'].values, subset['PC2'].values,
                                        edgecolor=colors_map.get(grp, 'gray'),
                                        facecolor=colors_map.get(grp, 'gray'),
                                        alpha=0.08, linewidth=2, linestyle='--')
        ax_l.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax_l.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax_l.set_title('PC1 vs PC2')
        ax_l.legend(fontsize=9)
        ax_l.grid(True, linestyle='--', alpha=0.6)
        ax_l.axhline(0, color='gray', linewidth=0.5)
        ax_l.axvline(0, color='gray', linewidth=0.5)
        ax_l.set_xlim(x_min, x_max)
        ax_l.set_ylim(y_min, y_max)
        # 重绘右图
        if n_components >= 3:
            for grp in [target_group, ref_group]:
                subset = scores_df[scores_df['Group'] == grp]
                ax_r.scatter(subset['PC1'], subset['PC3'],
                             label=f"{grp} (n={len(subset)})",
                             c=colors_map.get(grp, 'gray'), marker=markers_map.get(grp, 'o'),
                             s=80, alpha=0.7, edgecolors='black', linewidths=0.5)
            for grp in [target_group, ref_group]:
                subset = scores_df[scores_df['Group'] == grp]
                if len(subset) > 2:
                    plot_confidence_ellipse(ax_r, subset['PC1'].values, subset['PC3'].values,
                                            edgecolor=colors_map.get(grp, 'gray'),
                                            facecolor=colors_map.get(grp, 'gray'),
                                            alpha=0.08, linewidth=2, linestyle='--')
            ax_r.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax_r.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
            ax_r.set_title('PC1 vs PC3')
        else:
            for grp in [target_group, ref_group]:
                subset = scores_df[scores_df['Group'] == grp]
                ax_r.scatter(subset['PC1'], subset['PC2'],
                             label=f"{grp} (n={len(subset)})",
                             c=colors_map.get(grp, 'gray'), marker=markers_map.get(grp, 'o'),
                             s=80, alpha=0.7, edgecolors='black', linewidths=0.5)
            ax_r.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax_r.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            ax_r.set_title('PC1 vs PC2 (副本)')
        ax_r.legend(fontsize=9)
        ax_r.grid(True, linestyle='--', alpha=0.6)
        ax_r.axhline(0, color='gray', linewidth=0.5)
        ax_r.axvline(0, color='gray', linewidth=0.5)
        ax_r.set_xlim(x_min, x_max)
        ax_r.set_ylim(y_min, y_max)
        fig2.suptitle(f"PCA 得分图 (马氏距离 D_M = {D_M:.4f})", fontsize=14)
        fig2.tight_layout()
        fig2.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        img_buf.seek(0)
        st.download_button("📥 PCA 图 PNG",
                           img_buf, "PCA_Score_Plots.png", "image/png")

    st.success("🎉 计算完成！可以下载结果文件")

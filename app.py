import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import warnings
warnings.filterwarnings('ignore')

# ==================== 页面配置 ====================
st.set_page_config(page_title="PCA & 马氏距离分析", layout="wide")
st.title("📊 SEC-MS 相似性评估工具")
st.caption("上传 Excel → 自动 PCA 降维 → 马氏距离计算 → 一键下载报告")

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

c3, c4 = st.columns(2)
with c3:
    target_group = st.selectbox("目标组（如 DP）", unique_groups)
with c4:
    ref_default_idx = min(1, len(unique_groups) - 1) if len(unique_groups) > 1 else 0
    ref_group = st.selectbox("参考组（如 EU-RP）", unique_groups, index=ref_default_idx)

# ==================== 高级选项 ====================
with st.expander("⚙️ 高级选项（一般不需要改）"):
    c5, c6 = st.columns(2)
    with c5:
        var_thresh = st.slider("PCA 累计方差阈值", 0.80, 1.00, 0.99, 0.01)
    with c6:
        show_pair = st.checkbox("显示逐对距离明细", value=False)
    
    threshold = st.number_input("达标阈值", value=3.3, min_value=0.1, step=0.1)

# ==================== 开始计算 ====================
if st.button("🚀 开始计算", type="primary", use_container_width=True):
    # 识别数值列
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
    # 修改：保留 samples 为 pandas Series（后续用 iloc 索引）
    samples = df[sample_col]
    short_names = np.array([str(s).split()[-1] for s in samples])

    idx_target = np.where(groups == target_group)[0]
    idx_ref = np.where(groups == ref_group)[0]

    if len(idx_target) == 0 or len(idx_ref) == 0:
        st.error(f"❌ 目标组「{target_group}」或参考组「{ref_group}」没有匹配样本")
        st.stop()

    n_target = len(idx_target)
    n_ref = len(idx_ref)

    st.info(f"特征数: **{len(feature_cols)}** | {target_group}: **{n_target}** 个 | {ref_group}: **{n_ref}** 个")

    with st.spinner("🔄 PCA 降维 & 马氏距离计算中..."):
        # ---------- PCA ----------
        max_comp = min(X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=max_comp)
        X_pca = pca.fit_transform(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_)

        # 主成分数（严格避免秩亏）
        k = int(np.searchsorted(cumvar, var_thresh)) + 1
        k = min(k, n_ref - 2, n_target - 2, max_comp)
        k = max(k, 2)
        X_pca_k = X_pca[:, :k]

        # ---------- 马氏距离 ----------
        ref_data = X_pca_k[idx_ref]
        lw = LedoitWolf().fit(ref_data)
        inv_cov = lw.precision_
        mean_ref = ref_data.mean(axis=0)

        dists = []
        for i in idx_target:
            d = mahalanobis(X_pca_k[i], mean_ref, inv_cov)
            dists.append(d)

        # ---------- 逐对距离 ----------
        pair_rows = []
        if show_pair:
            for i in idx_target:
                for j in idx_ref:
                    d = mahalanobis(X_pca_k[i], X_pca_k[j], inv_cov)
                    pair_rows.append({
                        '目标样本': samples.iloc[i],  # 用 iloc 索引
                        '参考样本': samples.iloc[j],  # 用 iloc 索引
                        '马氏距离': round(d, 4)
                    })
        pair_df = pd.DataFrame(pair_rows) if pair_rows else None

        # ---------- PCA 图 ----------
        cmap = plt.cm.get_cmap('tab10')
        colors = {g: cmap(i) for i, g in enumerate(unique_groups)}
        markers = {g: ['o', 's', '^', 'D', 'v', 'P', '*'][i % 7] for i, g in enumerate(unique_groups)}

        fig1, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (pc1, pc2) in zip(axes, [(0, 1), (0, 2)]):
            for g in unique_groups:
                mask = groups == g
                ax.scatter(X_pca[mask, pc1], X_pca[mask, pc2],
                           c=[colors[g]], marker=markers[g],
                           label=g, s=80, alpha=0.8,
                           edgecolors='black', linewidths=0.5)
            for i in range(len(samples)):
                ax.annotate(short_names[i],
                            (X_pca[i, pc1], X_pca[i, pc2]),
                            fontsize=5.5, xytext=(2, 2), textcoords='offset points')
            ax.set_xlabel(f'PC{pc1+1} ({pca.explained_variance_ratio_[pc1]*100:.1f}%)')
            ax.set_ylabel(f'PC{pc2+1} ({pca.explained_variance_ratio_[pc2]*100:.1f}%)')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2, linestyle='--')
        fig1.tight_layout()

        # ---------- 结果表 ----------
        result_df = pd.DataFrame({
            # 修改：用 iloc 索引 pandas Series
            '样本名': samples.iloc[idx_target],
            '短名': short_names[idx_target],
            '马氏距离': [round(d, 4) for d in dists],
        })
        result_df['是否达标'] = result_df['马氏距离'].apply(
            lambda x: f'✅ 达标' if x < threshold else f'❌ 超标'
        )

        mean_d = np.mean(dists)
        max_d = np.max(dists)

    # ==================== 展示结果 ====================
    st.divider()
    st.subheader("📈 计算摘要")

    c_a, c_b, c_c, c_d = st.columns(4)
    c_a.metric("使用 PC 数", k)
    c_b.metric(f"PC1~PC{k} 累计方差", f"{cumvar[k-1]*100:.1f}%")
    c_a.metric("Mean 距离", f"{mean_d:.2f}",
               delta="✓达标" if mean_d < threshold else "✗超标",
               delta_color="normal" if mean_d < threshold else "inverse")
    c_d.metric("Max 距离", f"{max_d:.2f}",
               delta="✓达标" if max_d < threshold else "✗超标",
               delta_color="normal" if max_d < threshold else "inverse")

    st.subheader("🗺️ PCA 得分图")
    st.pyplot(fig1)
    plt.close(fig1)

    st.subheader("📏 马氏距离结果")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    if pair_df is not None:
        with st.expander(f"📊 {target_group} vs {ref_group} 逐对距离明细"):
            st.dataframe(pair_df, use_container_width=True, hide_index=True)

    with st.expander("📋 各主成分方差贡献"):
        pc_table = pd.DataFrame({
            '主成分': [f'PC{i+1}' for i in range(min(15, max_comp))],
            '方差贡献(%)': [f"{pca.explained_variance_ratio_[i]*100:.2f}" for i in range(min(15, max_comp))],
            '累计方差(%)': [f"{cumvar[i]*100:.2f}" for i in range(min(15, max_comp))],
        })
        st.dataframe(pc_table, use_container_width=True, hide_index=True)

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
            # 汇总
            summary_data = {
                '项目': ['目标组', '参考组', '目标样本数', '参考样本数',
                         '特征数', '使用PC数', f'PC1~PC{k}累计方差',
                         'Mean距离', 'Max距离', '达标阈值'],
                '值': [target_group, ref_group, n_target, n_ref,
                       len(feature_cols), k, f"{cumvar[k-1]*100:.1f}%",
                       round(mean_d, 4), round(max_d, 4), threshold]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='汇总', index=False)
            result_df.to_excel(writer, sheet_name='马氏距离', index=False)
            if pair_df is not None:
                pair_df.to_excel(writer, sheet_name='逐对距离', index=False)
        buf.seek(0)
        st.download_button("📥 完整报告 Excel",
                           buf, "PCA_mahalanobis_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with d_c3:
        img_buf = io.BytesIO()
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (pc1, pc2) in zip(axes2, [(0, 1), (0, 2)]):
            for g in unique_groups:
                mask = groups == g
                ax.scatter(X_pca[mask, pc1], X_pca[mask, pc2],
                           c=[colors[g]], marker=markers[g],
                           label=g, s=80, alpha=0.8,
                           edgecolors='black', linewidths=0.5)
            ax.set_xlabel(f'PC{pc1+1} ({pca.explained_variance_ratio_[pc1]*100:.1f}%)')
            ax.set_ylabel(f'PC{pc2+1} ({pca.explained_variance_ratio_[pc2]*100:.1f}%)')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2, linestyle='--')
        fig2.tight_layout()
        fig2.savefig(img_buf, format='png', dpi=200, bbox_inches='tight')
        plt.close(fig2)
        img_buf.seek(0)
        st.download_button("📥 PCA 图 PNG",
                           img_buf, "PCA_plot.png", "image/png")

    st.success("🎉 计算完成！可以下载结果文件")

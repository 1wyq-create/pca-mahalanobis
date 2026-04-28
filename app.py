import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== 页面配置 ====================
st.set_page_config(page_title="PCA & 马氏距离分析", layout="wide")
st.title("📊 SEC-MS 相似性评估工具")
st.caption("上传 Excel → 自动 PCA 降维 → 马氏距离计算")

# ==================== 侧边栏参数设置 ====================
with st.sidebar:
    st.header("⚙️ 参数设置")
    st.markdown("**上传文件后，在此选择列名和分组**")
    
    group_col = st.selectbox("分组列", options=[], index=0, key="group")
    sample_col = st.selectbox("样本名列", options=[], index=1, key="sample")
    
    st.divider()
    st.markdown("**选择目标组和参考组**")
    target_group = st.selectbox("目标组 (如 DP)", options=[], key="target")
    ref_group = st.selectbox("参考组 (如 EU-RP)", options=[], key="ref")
    
    st.divider()
    st.markdown("**高级选项**")
    var_thresh = st.slider("PCA 累计方差阈值", 0.80, 1.00, 0.99, 0.01)
    show_detail = st.checkbox("显示每个DP到每个RP的逐对距离", value=False)

# ==================== 文件上传 ====================
uploaded_file = st.file_uploader(
    "📁 点击或拖拽上传 Excel 文件", 
    type=['xlsx', 'xls'],
    key="file_uploader"
)

df = None

if uploaded_file is not None:
    # 读取所有 sheet 名
    xl = pd.ExcelFile(uploaded_file)
    
    # 尝试找 Data sheet，否则用第一个
    if 'Data' in xl.sheet_names:
        df = pd.read_excel(uploaded_file, sheet_name='Data')
        st.success(f"✅ 已读取 Data sheet，共 {df.shape[0]} 行 × {df.shape[1]} 列")
    else:
        df = pd.read_excel(uploaded_file, sheet_name=0)
        st.success(f"✅ 已读取第一个 sheet，共 {df.shape[0]} 行 × {df.shape[1]} 列")
    
    # 动态更新侧边栏选项
    cols = list(df.columns)
    st.session_state.group_options = cols
    st.session_state.sample_options = cols
    
    # 用 session_state 动态更新 selectbox
    # Streamlit 的 selectbox 不支持动态 options，需要用下面的 workaround
    st.session_state._cols = cols

# 用 rerun 机制实现动态下拉框
if 'file_uploader' in st.session_state and uploaded_file is not None:
    cols = st.session_state._cols
    
    # 重新渲染侧边栏（通过在 sidebar 里用 session_state 控制）
    if 'sidebar_inited' not in st.session_state:
        st.session_state.sidebar_inited = True
        st.session_state._group_default = next((i for i, c in enumerate(cols) if 'Group' in c or 'group' in c), 0)
        st.session_state._sample_default = next((i for i, c in enumerate(cols) if 'Sample' in c or 'sample' in c or 'Name' in c or 'name' in c), min(1, len(cols)-1))
        st.rerun()

# ==================== 实际处理逻辑 ====================
if uploaded_file is not None and df is not None:
    cols = list(df.columns)
    
    # 获取用户选择的列（因为 selectbox 动态更新有问题，这里直接读）
    # 用一种更简单的方式：让用户在上面直接输入
    with st.expander("📋 列名确认（如自动识别错误请手动修改）", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            group_col = st.selectbox("分组列", cols, index=st.session_state.get('_group_default', 0), key="g2")
        with col2:
            sample_col = st.selectbox("样本名列", cols, index=st.session_state.get('_sample_default', 1), key="s2")
    
    # 获取分组选项
    unique_groups = df[group_col].dropna().unique().tolist()
    
    col3, col4 = st.columns(2)
    with col3:
        target_idx = st.selectbox("目标组", unique_groups, key="t2")
    with col4:
        ref_idx = st.selectbox("参考组", unique_groups, index=min(1, len(unique_groups)-1), key="r2")
    
    # 排除非数值列
    feature_cols = [c for c in cols if c not in [group_col, sample_col] and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    if len(feature_cols) == 0:
        # 如果没识别到数值列，尝试排除前两列
        feature_cols = [c for c in cols if c not in [group_col, sample_col]]
        st.warning("⚠️ 未自动识别数值列，已默认排除前两列。请检查数据。")
    
    st.info(f"特征(桶)数: **{len(feature_cols)}** | 数据矩阵: **{df.shape[0]} 样本 × {len(feature_cols)} 特征**")
    
    # ========== 点击计算 ==========
    if st.button("🚀 开始计算", type="primary", use_container_width=True):
        X = df[feature_cols].values.astype(float)
        groups = df[group_col].values
        samples = df[sample_col].values
        
        idx_target = np.where(groups == target_idx)[0]
        idx_ref = np.where(groups == ref_idx)[0]
        
        if len(idx_target) == 0 or len(idx_ref) == 0:
            st.error("❌ 目标组或参考组没有匹配的样本！")
            st.stop()
        
        with st.spinner(" PCA 降维 & 马氏距离计算中..."):
            # -------- PCA --------
            max_comp = min(X.shape[0]-1, X.shape[1])
            pca = PCA(n_components=max_comp)
            X_pca = pca.fit_transform(X)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            
            # 确定使用的主成分数（严格限制避免秩亏）
            k = np.searchsorted(cumvar, var_thresh) + 1
            k = min(k, len(idx_ref) - 2, len(idx_target) - 2, max_comp)
            k = max(k, 2)
            X_pca_k = X_pca[:, :k]
            
            # -------- 马氏距离 --------
            ref_data = X_pca_k[idx_ref]
            lw = LedoitWolf().fit(ref_data)
            inv_cov = lw.precision_
            mean_ref = ref_data.mean(axis=0)
            
            dists = []
            for i in idx_target:
                d = mahalanobis(X_pca_k[i], mean_ref, inv_cov)
                dists.append(d)
            
            # 逐对距离
            pair_rows = []
            if show_detail:
                for i in idx_target:
                    for j in idx_ref:
                        d = mahalanobis(X_pca_k[i], X_pca_k[j], inv_cov)
                        pair_rows.append({
                            '目标样本': samples[i],
                            '参考样本': samples[j],
                            '马氏距离': round(d, 4)
                        })
            pair_df = pd.DataFrame(pair_rows) if pair_rows else None
            
            # -------- PCA 图 --------
            color_map = {}
            marker_map = {}
            cmap = plt.cm.get_cmap('tab10')
            for i, g in enumerate(unique_groups):
                color_map[g] = cmap(i)
                marker_map[g] = ['o', 's', '^', 'D', 'v', 'P', '*'][i % 7]
            
            fig1, axes = plt.subplots(1, 2, figsize=(14, 6))
            for ax, (pc1, pc2) in zip(axes, [(0,1), (0,2)]):
                for g in unique_groups:
                    mask = groups == g
                    ax.scatter(X_pca[mask, pc1], X_pca[mask, pc2],
                               c=[color_map[g]], marker=marker_map[g],
                               label=g, s=80, alpha=0.8,
                               edgecolors='black', linewidths=0.5)
                for i in range(len(samples)):
                    ax.annotate(str(samples[i]).split()[-1], 
                                (X_pca[i, pc1], X_pca[i, pc2]),
                                fontsize=5.5, xytext=(2, 2), textcoords='offset points')
                ax.set_xlabel(f'PC{pc1+1} ({pca.explained_variance_ratio_[pc1]*100:.1f}%)')
                ax.set_ylabel(f'PC{pc2+1} ({pca.explained_variance_ratio_[pc2]*100:.1f}%)')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.2, linestyle='--')
            fig1.tight_layout()
            
            # -------- 结果表 --------
            result_df = pd.DataFrame({
                '样本名': samples[idx_target],
                '马氏距离': [round(d, 4) for d in dists]
            })
            result_df['是否达标(<3.3)'] = result_df['马氏距离'].apply(lambda x: '✅ 达标' if x < 3.3 else '❌ 超标')
            
            summary = {
                '目标组': target_idx,
                '参考组': ref_idx,
                '样本数': f"{len(idx_target)} vs {len(idx_ref)}",
                '特征数': len(feature_cols),
                '使用PC数': k,
                f'PC1~{k}累计方差': f"{cumvar[k-1]*100:.1f}%",
                'Mean距离': round(np.mean(dists), 4),
                'Max距离': round(np.max(dists), 4),
            }
        
        # ========== 展示结果 ==========
        st.divider()
        st.subheader("📈 计算摘要")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("使用 PC 数", summary['使用PC数'])
        col_b.metric("累计方差", summary[f'PC1~{k}累计方差'])
        col_c.metric("Mean 距离", summary['Mean距离'], 
                      delta="达标 ✓" if summary['Mean距离'] < 3.3 else "超标 ✗",
                      delta_color="normal" if summary['Mean距离'] < 3.3 else "inverse")
        col_d.metric("Max 距离", summary['Max距离'],
                      delta="达标 ✓" if summary['Max距离'] < 3.3 else "超标 ✗",
                      delta_color="normal" if summary['Max距离'] < 3.3 else "inverse")
        
        st.subheader("🗺️ PCA 得分图")
        st.pyplot(fig1)
        plt.close(fig1)
        
        st.subheader("📏 马氏距离结果")
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        
        if pair_df is not None:
            with st.expander("查看逐对距离明细"):
                st.dataframe(pair_df, use_container_width=True, hide_index=True)
        
        # PC 方差贡献表
        with st.expander("查看各主成分方差贡献"):
            pc_table = pd.DataFrame({
                '主成分': [f'PC{i+1}' for i in range(min(15, max_comp))],
                '方差贡献(%)': [f"{pca.explained_variance_ratio_[i]*100:.2f}" for i in range(min(15, max_comp))],
                '累计方差(%)': [f"{cumvar[i]*100:.2f}" for i in range(min(15, max_comp))],
            })
            st.dataframe(pc_table, use_container_width=True, hide_index=True)
        
        # 下载按钮
        col_x, col_y = st.columns(2)
        with col_x:
            csv = result_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下载马氏距离结果", csv, "mahalanobis_results.csv", "text/csv")
        with col_y:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                result_df.to_excel(writer, sheet_name='汇总', index=False)
                if pair_df is not None:
                    pair_df.to_excel(writer, sheet_name='逐对距离', index=False)
            buf.seek(0)
            st.download_button("📥 下载 Excel 完整报告", buf, "report.xlsx", 
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

import io  # 放在文件末尾即可



import pandas as pd
import numpy as np
from tigramite.pcmci import PCMCI
from tigramite import plotting
from tigramite.independence_tests import gpdc, cmiknn, CMIsymb, ParCorr
import tigramite.data_processing as pp
from tigramite.data_processing import DataFrame
from matplotlib import pyplot as plt
from tigramite import plotting as tp

# 读取数据
df = pd.read_csv(r'C:\Users\7iCK\Desktop\10_features.csv')

data = df.values
tdf = DataFrame(
    data=data,
    mask=None,
    missing_flag=None,
    var_names=df.columns,
    datatime=None
)

# 设置条件独立性检验
cond_ind_test = ParCorr()

# 进行 PCMCI 分析
pcmci = PCMCI(dataframe=tdf, cond_ind_test=cond_ind_test)
results = pcmci.run_pcmciplus(tau_min=0, tau_max=0, pc_alpha=None)
pcmci.print_results(results, alpha_level=0.01)

# 绘制滞后函数
correlations = pcmci.get_lagged_dependencies(tau_max=0, val_only=True)['val_matrix']
lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, setup_args={'var_names': df.columns,
                                                                        'x_base': 5, 'y_base': .5})
plt.show()

# 获取校正后的 p 值矩阵
q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=0, fdr_method='fdr_bh')
link_matrix = pcmci.return_significant_links(pq_matrix=q_matrix,
                                             val_matrix=results['val_matrix'], alpha_level=0.01)['link_matrix']

# 绘制图形
tp.plot_graph(
    val_matrix=results['val_matrix'],
    link_matrix=link_matrix,
    var_names=df.columns,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI'
)
plt.show()

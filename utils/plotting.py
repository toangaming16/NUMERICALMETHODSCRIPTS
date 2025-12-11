import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
def create_plot(df_results, title):
    """
    Tạo một đồ thị Matplotlib so sánh tất cả các giải pháp.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'y_Exact' in df_results.columns:
        ax.plot(df_results['t'], df_results['y_Exact'], 'k-', label='y_Exact (Giải tích)', linewidth=2, zorder=10)
    for col in df_results.columns:
        if col.startswith('y_') and col != 'y_Exact':
            ax.plot(df_results['t'], df_results[col], 'o-', label=col, markersize=4, alpha=0.8)
        elif col.startswith('Error_'):
            pass 
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Biến độc lập (t)', fontsize=12)
    ax.set_ylabel('Biến phụ thuộc (y)', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle=':')
    return fig
def create_results_dataframe(t_values, solutions, exact_func=None):
    """
    Tạo một DataFrame để tóm tắt và phân tích lỗi.
    solutions là một dict: {'y_Taylor': y_taylor_data,...}
    """
    df = pd.DataFrame({'t': t_values})
    for name, data in solutions.items():
        if len(data) == len(t_values):
            df[name] = data
    if exact_func:
        try:
            y_exact = exact_func(t_values)
            df['y_Exact'] = y_exact
            for name, data in solutions.items():
                if len(data) == len(t_values):
                    if name.startswith('y_') and name != 'y_Exact':
                        df[f'Error_{name}'] = np.abs(data - y_exact)
        except Exception as e:
            st.warning(f"Không thể tính toán giải pháp giải tích: {e}")  
    return df
import numpy as np
import pandas as pd
import streamlit as st
class RKF45Solver:
    def __init__(self, f_numeric):
        self.f = f_numeric
    def solve(self, t0, y0, tend, N):
        t_values = np.linspace(t0, tend, N + 1)
        y_values = np.zeros(N + 1)      
        y_values[0] = y0   
        h = (tend - t0) / N
        step_data = [] 
        step_explanations = []
        columns = ["Step", "t_i", "y_i", "k1", "k2", "k3", "k4", "k5", "k6", "y_next"]
        for i in range(N):
            ti = t_values[i]
            yi = y_values[i]
            step_str = f"**Bước {i+1} (từ t = {ti:.4f} đến t = {t_values[i+1]:.4f}):**\n\n"
            step_str += f"* Giá trị ban đầu: $y_{i} = y({ti:.4f}) = {yi:.6f}$\n"
            step_str += f"* $h = {h:.4f}$\n"
            step_str += "* Tính 6 hệ số $k$:\n"
            k1 = h * np.real(self.f(ti, yi))
            step_str += f"    * $k_1 = h \cdot f(t_i, y_i) = {h:.4f} \cdot f({ti:.4f}, {yi:.6f}) = {k1:.6f}$\n"
            k2 = h * np.real(self.f(ti + h/4, yi + k1/4))
            step_str += f"    * $k_2 = h \cdot f(t_i + h/4, y_i + k_1/4) = {k2:.6f}$\n"
            k3 = h * np.real(self.f(ti + 3*h/8, yi + 3*k1/32 + 9*k2/32))
            step_str += f"    * $k_3 = h \cdot f(t_i + 3h/8, y_i + 3k_1/32 + 9k_2/32) = {k3:.6f}$\n" 
            k4 = h * np.real(self.f(ti + 12*h/13, yi + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197))
            step_str += f"    * $k_4 = h \cdot f(t_i + 12h/13, y_i + \dots) = {k4:.6f}$\n"
            k5 = h * np.real(self.f(ti + h, yi + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104))
            step_str += f"    * $k_5 = h \cdot f(t_i + h, y_i + \dots) = {k5:.6f}$\n"
            k6 = h * np.real(self.f(ti + h/2, yi - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40))
            step_str += f"    * $k_6 = h \cdot f(t_i + h/2, y_i + \dots) = {k6:.6f}$\n"
            y_next = yi + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6
            step_str += f"\n* Cộng các hệ số (công thức bậc 5):\n"
            step_str += f"    $y_{i+1} = y_i + \\frac{{16}}{{135}}k_1 + \\frac{{6656}}{{12825}}k_3 + \dots$\n"
            step_str += f"    **$y_{i+1} = {y_next:.6f}$**\n\n---\n"
            y_values[i+1] = y_next
            step_data.append([i, ti, yi, k1, k2, k3, k4, k5, k6, y_next])
            step_explanations.append(step_str)
        with st.expander("Xem chi tiết tính toán từng bước của RKF45"):
            tab1, tab2 = st.tabs(["Giải thích từng bước", "Bảng dữ liệu chi tiết"])
            with tab1:
                if step_data:
                    tab1.markdown("\n".join(step_explanations))
                else:
                    tab1.info("Không có dữ liệu bước để hiển thị.")
            with tab2:
                if step_data:
                    df_steps = pd.DataFrame(step_data, columns=columns)
                    tab2.dataframe(df_steps.style.format("{:.6f}"))
                else:
                    tab2.info("Không có dữ liệu bước để hiển thị.")
        return t_values, y_values
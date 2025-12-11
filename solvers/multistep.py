import numpy as np
import pandas as pd
import streamlit as st
from rk import RKF45Solver
class ABM4Solver:
    def __init__(self, f_numeric):
        self.f = f_numeric

    def solve(self, t0, y0, tend, N):
        if N < 4:
            return np.array([]), np.array([])
        t_values = np.linspace(t0, tend, N + 1)
        y_values = np.zeros(N + 1)        
        y_values[0] = y0  
        h = (tend - t0) / N
        rk_solver = RKF45Solver(self.f)
        _, y_startup = rk_solver.solve(t0, y0, t0 + 3*h, 3)
        y_values[0:4] = np.real(y_startup)
        f_values = np.zeros(N + 1)
        for i in range(4):
            f_values[i] = np.real(self.f(t_values[i], y_values[i]))   
        step_data = [] 
        step_explanations = []
        columns = ["Step (i)", "t_i", "y_i", "y_pred", "f_pred", "y_new"]
        for i in range(3, N):
            ti = t_values[i]
            yi = y_values[i]
            step_str = f"**Bước {i+1} (từ t = {ti:.4f} đến t = {t_values[i+1]:.4f}):**\n\n"
            step_str += f"* Các giá trị $f$ đã biết:\n"
            step_str += f"    * $f_{i} = f({t_values[i]:.4f}, {y_values[i]:.6f}) = {f_values[i]:.6f}$\n"
            step_str += f"    * $f_{i-1} = f({t_values[i-1]:.4f}, {y_values[i-1]:.6f}) = {f_values[i-1]:.6f}$\n"
            step_str += f"    * $f_{i-2} = f({t_values[i-2]:.4f}, {y_values[i-2]:.6f}) = {f_values[i-2]:.6f}$\n"
            step_str += f"    * $f_{i-3} = f({t_values[i-3]:.4f}, {y_values[i-3]:.6f}) = {f_values[i-3]:.6f}$\n"
            p_next = yi + (h/24) * (
                55 * f_values[i] 
                - 59 * f_values[i-1] 
                + 37 * f_values[i-2] 
                - 9 * f_values[i-3]
            )
            step_str += f"\n* **(P) Dự đoán (Predictor):**\n"
            step_str += f"    $p_{i+1} = y_i + \\frac{{h}}{{24}} (55f_i - 59f_{i-1} + 37f_{i-2} - 9f_{i-3})$\n"
            step_str += f"    $p_{i+1} = {yi:.6f} + \\frac{{{h:.4f}}}{{24}} (55 \cdot ({f_values[i]:.6f}) - \dots) = {p_next:.6f}$\n"
            t_next = t_values[i+1]
            f_predicted = np.real(self.f(t_next, p_next))
            step_str += f"\n* **(E) Đánh giá (Evaluate):**\n"
            step_str += f"    $f^p_{i+1} = f(t_{i+1}, p_{i+1}) = f({t_next:.4f}, {p_next:.6f}) = {f_predicted:.6f}$\n"
            y_next = yi + (h/24) * (
                9 * f_predicted     
                + 19 * f_values[i]
                - 5 * f_values[i-1]
                + 1 * f_values[i-2]
            )
            step_str += f"\n* **(C) Hiệu chỉnh (Corrector):**\n"
            step_str += f"    $y_{i+1} = y_i + \\frac{{h}}{{24}} (9f^p_{i+1} + 19f_i - 5f_{i-1} + f_{i-2})$\n"
            step_str += f"    $y_{i+1} = {yi:.6f} + \\frac{{{h:.4f}}}{{24}} (9 \cdot ({f_predicted:.6f}) + \dots)$\n"
            step_str += f"    **$y_{i+1} = {y_next:.6f}$**\n\n---\n"
            y_values[i+1] = y_next
            f_values[i+1] = np.real(self.f(t_next, y_next))
            step_data.append([i, ti, yi, p_next, f_predicted, y_next])
            step_explanations.append(step_str)
        with st.expander("Xem chi tiết tính toán từng bước của ABM4 (Predictor-Corrector)"):
            tab1, tab2 = st.tabs(["Giải thích từng bước", "Bảng dữ liệu chi tiết"]) 
            with tab1:
                tab1.markdown(f"**Lưu ý:** 3 bước đầu tiên (để có $y_1, y_2, y_3$) được tính tự động bằng RKF45 để khởi động.")
                if step_data:
                    tab1.markdown("\n".join(step_explanations))
                else:
                    tab1.info("Chưa đủ dữ liệu để hiển thị bước ABM4 (cần N >= 4).")

            with tab2:
                tab2.markdown(f"**Lưu ý:** 3 bước đầu tiên (để có $y_1, y_2, y_3$) được tính tự động bằng RKF45 để khởi động.")
                if step_data:
                    df_steps = pd.DataFrame(step_data, columns=columns)
                    tab2.dataframe(df_steps.style.format("{:.6f}"))
                else:
                    tab2.info("Chưa đủ dữ liệu để hiển thị bước ABM4 (cần N >= 4).")

        return t_values, y_values
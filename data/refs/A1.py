# A1: формула Байеса. P(D)=0.01, P(+|D)=0.95, P(+|не D)=0.05. Найти P(D|+).
p_d, p_pos_d, p_pos_nd = 0.01, 0.95, 0.05
p_pos = p_pos_d * p_d + p_pos_nd * (1 - p_d)   # полная вероятность срабатывания
print(f"REFERENCE: {p_pos_d * p_d / p_pos:.4f}")

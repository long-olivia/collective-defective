import json

with open("gc_rounds.json") as file:
    gc_rounds = json.load(file)

with open("self_gc_rounds.json") as file:
    self_gc_rounds = json.load(file)

with open("lq_rounds.json") as file:
    lq_rounds = json.load(file)

with open("self_lq_rounds.json") as file:
    self_lq_rounds = json.load(file)

gc_avg_diff={"CC": [0,0], "CN": [0,0], "CS": [0,0], "NC": [0,0], "NN": [0,0], "NS": [0,0], "SC": [0,0], "SN": [0,0], "SS": [0,0]}
lq_avg_diff={"CC": [0,0], "CN": [0,0], "CS": [0,0], "NC": [0,0], "NN": [0,0], "NS": [0,0], "SC": [0,0], "SN": [0,0], "SS": [0,0]}

for pairing in gc_rounds:
    g_diff=[0]*20
    c_diff=[0]*20
    l_diff=[0]*20
    q_diff=[0]*20
    for i in range(0,20):
        g_diff[i] = self_gc_rounds[pairing][0][i] - gc_rounds[pairing][0][i]
        c_diff[i] = self_gc_rounds[pairing][1][i] - gc_rounds[pairing][1][i]
        l_diff[i] = self_lq_rounds[pairing][0][i] - lq_rounds[pairing][0][i]
        q_diff[i] = self_lq_rounds[pairing][1][i] - lq_rounds[pairing][1][i]
    g_avg = float(sum(g_diff)/20)
    c_avg = float(sum(c_diff)/20)
    l_avg = float(sum(l_diff)/20)
    q_avg = float(sum(q_diff)/20)
    gc_avg_diff[pairing][0] = g_avg
    gc_avg_diff[pairing][1] = c_avg
    lq_avg_diff[pairing][0] = l_avg
    lq_avg_diff[pairing][1] = q_avg

with open("gc_avg_diff.json", 'w') as f:
    json.dump(gc_avg_diff, f) 

with open("lq_avg_diff.json", 'w') as f:
    json.dump(lq_avg_diff, f)
import math
import random

# البيانات
data = ["I", "love", "deep", "learning"]
word_to_ix = {word: i for i, word in enumerate(data)}
ix_to_word = {i: word for word, i in word_to_ix.items()}

# تحويل إلى one-hot
def one_hot(index, size):
    vec = [0] * size
    vec[index] = 1
    return vec

# إعدادات الشبكة
vocab_size = len(data)
hidden_size = 6
learning_rate = 0.1

# تهيئة الأوزان عشوائيًا
def rand_matrix(rows, cols):
    return [[random.uniform(-0.1, 0.1) for _ in range(cols)] for _ in range(rows)]

Wxh = rand_matrix(hidden_size, vocab_size)
Whh = rand_matrix(hidden_size, hidden_size)
Why = rand_matrix(vocab_size, hidden_size)
bh = [0] * hidden_size
by = [0] * vocab_size

# دوال مساعدة
def tanh(x):
    return [math.tanh(i) for i in x]

def dtanh(x):
    return [1 - math.tanh(i) ** 2 for i in x]

def softmax(x):
    max_x = max(x)
    exps = [math.exp(i - max_x) for i in x]
    sum_exps = sum(exps)
    return [j / sum_exps for j in exps]

# تمرير أمامي وعكسي
def forward_backward(inputs, target_index):
    xs, hs, h_inputs = {}, {}, {}
    hs[-1] = [0] * hidden_size
    ys, ps = {}, {}

    # التمرير الأمامي
    for t in range(len(inputs)):
        xs[t] = one_hot(inputs[t], vocab_size)
        h_input = [0] * hidden_size
        for i in range(hidden_size):
            sum_in = sum(Wxh[i][j] * xs[t][j] for j in range(vocab_size))
            sum_h = sum(Whh[i][j] * hs[t-1][j] for j in range(hidden_size))
            h_input[i] = sum_in + sum_h + bh[i]
        h_inputs[t] = h_input
        hs[t] = tanh(h_input)

    # الإخراج
    y = [0] * vocab_size
    for i in range(vocab_size):
        y[i] = sum(Why[i][j] * hs[len(inputs)-1][j] for j in range(hidden_size)) + by[i]
    ps = softmax(y)

    # الخسارة
    loss = -math.log(ps[target_index])

    # التمرير العكسي
    dWxh = [[0]*vocab_size for _ in range(hidden_size)]
    dWhh = [[0]*hidden_size for _ in range(hidden_size)]
    dWhy = [[0]*hidden_size for _ in range(vocab_size)]
    dbh = [0]*hidden_size
    dby = [0]*vocab_size
    dh_next = [0]*hidden_size

    # تدرج الإخراج
    dy = ps[:]
    dy[target_index] -= 1

    for i in range(vocab_size):
        for j in range(hidden_size):
            dWhy[i][j] += dy[i] * hs[len(inputs)-1][j]
        dby[i] += dy[i]

    # تدرج الحالة الخفية
    for t in reversed(range(len(inputs))):
        dh = [0]*hidden_size
        for j in range(hidden_size):
            for i in range(vocab_size):
                dh[j] += dy[i] * Why[i][j]
            dh[j] += dh_next[j]

        dh_raw = [dh[i] * dtanh([h_inputs[t][i]])[0] for i in range(hidden_size)]

        for i in range(hidden_size):
            for j in range(vocab_size):
                dWxh[i][j] += dh_raw[i] * xs[t][j]
            for j in range(hidden_size):
                dWhh[i][j] += dh_raw[i] * hs[t-1][j]
            dbh[i] += dh_raw[i]

        dh_next = [sum(Whh[j][i] * dh_raw[j] for j in range(hidden_size)) for i in range(hidden_size)]

    # تحديث الأوزان
    for i in range(hidden_size):
        for j in range(vocab_size):
            Wxh[i][j] -= learning_rate * dWxh[i][j]
    for i in range(hidden_size):
        for j in range(hidden_size):
            Whh[i][j] -= learning_rate * dWhh[i][j]
    for i in range(vocab_size):
        for j in range(hidden_size):
            Why[i][j] -= learning_rate * dWhy[i][j]
    for i in range(hidden_size):
        bh[i] -= learning_rate * dbh[i]
    for i in range(vocab_size):
        by[i] -= learning_rate * dby[i]

    return loss, ps

# التدريب
inputs = [word_to_ix["I"], word_to_ix["love"], word_to_ix["deep"]]
target = word_to_ix["learning"]

for epoch in range(100):
    loss, pred = forward_backward(inputs, target)
    if epoch % 10 == 0:
        pred_word = ix_to_word[pred.index(max(pred))]
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Prediction: {pred_word}")

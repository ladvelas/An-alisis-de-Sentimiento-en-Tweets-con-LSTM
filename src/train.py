import argparse

def train_rnn():
    print("Entrenando modelo RNN...")

def train_lstm():
    print("Entrenando modelo LSTM...")

def train_bilstm_attention():
    print("Entrenando modelo BiLSTM con atención...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    if args.model == "rnn":
        train_rnn()
    elif args.model == "lstm":
        train_lstm()
    elif args.model == "bilstm_attention":
        train_bilstm_attention()
    else:
        print(f"Modelo '{args.model}' no reconocido.")


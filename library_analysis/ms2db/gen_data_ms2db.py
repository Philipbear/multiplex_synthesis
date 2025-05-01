import pickle


def main():
    with open('all_ms2db.pkl', 'rb') as f:
        ms2db = pickle.load(f)

    print(f"Loaded {len(ms2db)} spectra from all_ms2db.pkl")



if __name__ == '__main__':
    main()



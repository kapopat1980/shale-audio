import pickle
for n in ('crema', 'vctk'):
    r = pickle.load(open(f'results/real_{n}_0.pkl', 'rb')) + pickle.load(open(f'results/real_{n}_1.pkl', 'rb'))
    pickle.dump(r, open(f'results/real_{n}.pkl', 'wb')); print(n, len(r))

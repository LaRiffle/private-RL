import mdptoolbox.example
P, R = mdptoolbox.example.small()

print(P)
print(R)

vi = mdptoolbox.mdp.ValueIteration(P, R, 0.96)
vi.setVerbose()
vi.run()
print('Optimal value function: {}'.format(vi.V))
print('Optimal policy: {}'.format(vi.policy))

# ql = mdptoolbox.mdp.QLearning(P, R, 0.96)
# ql.setVerbose()
# ql.run()
# print('qlearning q: {}'.format(ql.Q))
# print('qlearning policy: {}'.format(ql.policy))
from reasoners.visualization import TreeLog
import os 
import pickle

def node_visualizer(x):
    if not x.state:
        return {}
    # return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}
    # match states
    if x.state[-1].state_type == "ANSWER":
        return {"question": x.state[-1].main_question, "answer": x.state[-1].main_answer}
    elif x.state[-1].state_type == "RETRIEVE":
        return {"context": x.state[-1].context, "retrieved_snippets": x.state[-1].retrieved_snippets}


with open('/home/stud/abinder/reference_implementations/llm-reasoners/examples/rap_webqa/logs/gsm8k_MCTS/10282023-125829/algo_output/1319.pkl', 'rb') as f:
    algo_output = pickle.load(f)

with open('/home/stud/abinder/reference_implementations/llm-reasoners/examples/rap_webqa/test.json', 'w') as f:
    # noinspection PyTypeChecker
    print(TreeLog.from_mcts_results(algo_output, node_data_factory=node_visualizer), file=f)

import re
from typing import Optional, Union

import io

def load_webqa_dataset(path_to_webqa, split, resume):
    import json
    from pathlib import Path

    with open(Path(path_to_webqa), 'r') as file:
        data = json.load(file)
        
    if isinstance(resume, str):
        start_str, end_str = resume.strip('[]').split(',')
        start, end = int(start_str), int(end_str)
        selected_indices = list(range(start, end+1))
    if isinstance(resume, int):
        selected_indices = [resume]
    if isinstance(resume, list):
        selected_indices = list(range(resume[0], resume[1]+1))
    
    selected_data = [
        {
            "index": idx, 
            "question": data[idx]["Q"], 
            "Guid": data[idx]["Guid"], 
            "ground-truth": data[idx]["A"],
            "Qcate": data[idx]["Qcate"],
            "path": path_to_webqa
        } 
        for idx in selected_indices if idx < len(data)
    ]
    return selected_data


def find_first_appearance(text, keys):
    keywords = list(keys)
    
    for keyword in keywords:
        if keyword in text:
            return keyword
    
    return None 
   

def action_selection_prompt(prompt, example, state):
    with io.StringIO() as f:
        f.write(prompt["action_selection"]["description"] + "\n") 

        # give overall question
        f.write(prompt["general"]["prefix_main"] + example + "\n") 

        # write action descriptions
        f.write(prompt["action_selection"]["options"] + "\n") 
        for idx, a in enumerate(prompt["actions"]):
            f.write(a + ": " + prompt["actions"][a]["description"] + "\n")

        # write history
        # only write if history exists
        if state != []:
            f.write(prompt["action_selection"]["history"] + "\n") 
            for idx, a in enumerate(prompt["actions"]):
                # do not print action with no history
                if any(s.state_type == a for s in state):
                    # f.write(a + ": " + "\n")
                    # for idx, s in enumerate(state):
                    #     if a == s.state_type:
                    #         #g.write(s[0] + " "+ s[1] + "\n")
                    #         f.write(get_history(s, a) + "\n")
                    f.write(get_history(state, a)) # TODO 

        # output format
        f.write(prompt["action_selection"]["output_format"] + "\n") 
        model_input = f.getvalue()
    return model_input

def action_prompt(prompt, example, state, action):
    
    with io.StringIO() as g:
        g.write(prompt["actions"][action]["description"] + "\n") 

        # give overall question
        g.write(prompt["general"]["prefix_main"] + example + "\n") 

        # write history
        # only write if history exists
        if state != []:
            g.write(prompt["actions"][action]["history"] + "\n") 
            for idx, a in enumerate(prompt["actions"]):
                # do not print action with no history
                if any(s.state_type == a for s in state):
                    #g.write(a + ": " + "\n")
                    # TODO long term solution should be cleaner
                    # for idx, s in enumerate(state):
                    #     if a == s.state_type:
                    #         #g.write(s[0] + " "+ s[1] + "\n")
                    #         g.write(get_history(s, a) + "\n")
                    # TODO right now anyway only retrieve has history
                    g.write(get_history(state, a)) # TODO 

        # # write examples
        # g.write(prompt["actions"][action]["examples"]["prefix"] + "\n") 
        # for idx, (parent_question, child_question) in enumerate(prompt["actions"][action]["examples"]["data"]):

        #     g.write("Parent Question: " + prompt["actions"][action]["examples"]["data"][idx]["parent_question"] + "\n")
        #     g.write("Child Question: " + prompt["actions"][action]["examples"]["data"][idx]["child_question"] + "\n")

        #g.write("Please write a query that uses the context to get new information.")

        # output format
        g.write(prompt["actions"][action]["output_format"] + "\n") 
        # g.write("Parent Question: " + example + "\n")
        # g.write("Child Question: ")
        model_input = g.getvalue()
    return model_input      

def evaluation_prompt(action_prompt, prompt, example, state, action):
    
    # old code
    # with io.StringIO() as f:
    #         f.write(self.useful_prompt["input"])
    #         f.write(self.useful_prompt["question_prefix"] + self.example + "\n")
    #         for idx, (q, _, _) in enumerate(state):
    #             f.write(self.useful_prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
    #         f.write(self.useful_prompt["new_subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")
    #         f.write(self.useful_prompt["useful_prefix"])
    #         model_input = f.getvalue()

    with io.StringIO() as g:
        g.write(prompt["general"]["description"] + "\n") 

        # give overall question
        g.write(prompt["general"]["prefix_main"] + example + "\n") 

        # write history
        # only write if history exists
        if state != []:
            g.write(prompt["general"]["history"] + "\n") 
            for idx, a in enumerate(action_prompt["actions"]):
                # do not print action with no history
                if any(s.state_type == a for s in state):
                    #g.write(a + ": " + "\n")
                    # for idx, s in enumerate(state):
                    #     if a == s.state_type:
                    #         #g.write(s[0] + " "+ s[1] + "\n")
                    #         g.write(get_history(s, a) + "\n")
                    g.write(get_history(state, a)) # TODO 

        # # write examples
        # g.write(prompt["actions"][action]["examples"]["prefix"] + "\n") 
        # for idx, (parent_question, child_question) in enumerate(prompt["actions"][action]["examples"]["data"]):

        #     g.write("Parent Question: " + prompt["actions"][action]["examples"]["data"][idx]["parent_question"] + "\n")
        #     g.write("Child Question: " + prompt["actions"][action]["examples"]["data"][idx]["child_question"] + "\n")

        # output format
        g.write(prompt["general"]["prefix_action"] + " " + action + "\n") 
        # suffix_action
        g.write(prompt["general"]["suffix_action"] + "\n")  
        
        model_input = g.getvalue()
    return model_input
        
def answer_prompt(prompt, example, state, action):

    with io.StringIO() as g:
        g.write(prompt["actions"][action]["description"] + "\n") 

        # give overall question
        g.write(prompt["general"]["prefix_main"] + example + "\n") 

        # write history
        # only write if history exists
        if state != []:
            g.write(prompt["actions"][action]["history"] + "\n") 
            for idx, a in enumerate(prompt["actions"]):
                # do not print action with no history
                if any(s.state_type == a for s in state):
                    #g.write(a + ": " + "\n")
                    # for idx, s in enumerate(state):
                    #     if a == s.state_type:
                    #         #g.write(s[0] + " "+ s[1] + "\n")
                    #         g.write(get_history(s, a) + "\n")
                    g.write(get_history(state, a)) # TODO 

        # output format
        g.write(prompt["actions"][action]["output_format"] + "\n") 
        # g.write("Parent Question: " + example + "\n")
        # g.write("Child Question: ")
        model_input = g.getvalue()
    return model_input      

def get_history(state_list, action):
    # if action == "RETRIEVE":
    #     return state.retrieved_snippets
    snippets = [snippet for state in state_list for snippet in state.retrieved_snippets]
    out = ""
    for idx, snippet in enumerate(snippets):
        out += f"{idx}) {snippet}" + "\n"
    return out
    


def retrieve_answer(output: Union[list, str]) -> Optional[str]:
    '''
    output should be a world_model.GSM8kState if being a list
    '''
    if isinstance(output, list):
        output = output[-1].sub_answer
    match = re.match(r'.*The answer is .*?([ $.0-9,\-]+).*\..*', output)
    if match is None:
        return None
    answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
    if '=' in answer:
        answer = answer[answer.rindex('=') + 1:]
    return answer


def retrieve_answer_from_dataset(answer: str) -> str:
    return re.match(r'[\S\s]*#### (.*)$', answer)[1]


def judge_answer(output: Optional[str], answer: str) -> bool:
    if output is None:
        return False
    try:
        output = int(output)
        answer = int(answer)
        return output == answer
    except ValueError:
        pass
    try:
        output = float(output)
        answer = float(answer)
        return output == answer
    except ValueError:
        pass
    return output == answer

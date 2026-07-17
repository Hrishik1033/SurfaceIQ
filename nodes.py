from typing import List

from langchain_classic.schema import prompt_template
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

llm = ChatMistralAI(model = 'mistral-small-2506')
class State(BaseModel):
    defect_names:list[str] = Field(description="It stores the list of defect names")
    description:list[str] = Field(description="It stores the detailed description of the defect")
    steps_to_correct:list[str] = Field(description="It stores the steps taken to correct the defect of rectify it")
    financial_feasibility:list[str] = Field(description="It stores the financial feasibility of the defect,whether the defect is financially feasible to cure")


def defect_description_node(state)->State:
    """It provides a detailed description of the defect present"""
    prompt_template = ChatPromptTemplate.from_messages([
        ('system',' You are a defect description expert. You will be provided with the defect name and you have to provide a detailed description of the defect, steps to correct it and financial feasibility of the defect.'),
        ('human','{defect}')
    ])
    chain = prompt_template | llm
    for defect in state.defect_names:
        result = chain.invoke({'defect':defect})
        state.description.append(result.content)
    return state


def steps_to_correct_node(state)->State:
    """It provides the steps to correct the defect present in a well-structured manner"""
    prompt_template = ChatPromptTemplate.from_messages([
        ('system',' You are a defect correction expert. You will be provided with the defect name and you have to provide the steps to correct the defect.Provide a well structured defect correction path in a arrow based format where each step is succeeded by an arrow.'),
        ('human','{defect}')
    ])
    chain = prompt_template | llm
    for defect in state.defect_names:
        result = chain.invoke({'defect':defect})
        state.steps_to_correct.append(result.content)
    return state


def financial_feasibility_node(state)->State:
    """It provides the financial feasibility of the defect present"""
    prompt_template = ChatPromptTemplate.from_messages([
        ('system',' You are a financial feasibility expert. You will be provided with the defect name and you have to provide the financial feasibility of the defect,whether the defect is financially feasible to cure or not ie. whether correcting the defect will be cost effective or not.'),
        ('human','{defect}')
    ])
    chain = prompt_template | llm
    for defect in state.defect_names:
        result = chain.invoke({'defect':defect})
        state.financial_feasibility.append(result.content)
    return state

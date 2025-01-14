TestAgent: This agent creates testing scenarios for chatbot.

Based on existing agents and known intents to them, it creates testing scenarios

**Descriptions**: 
1. put the number to choose the query and agent like "select data from database" - mysql_agent
2. put the amount scenarios u want to get
3. result is put in file 'scenarios/scenarios_{agent_name}_{H-M-S_d-m-Y}.json'
4. agent_test.py - code of the agent to create tests
5. get_test.py - runnable file to generate scenarios

**Output format**:

{

    "queries": [
        {
            "input": "",
            "expected_intent": "",
            "expected_agent": "",
            "expected_output": ""
        }
    ]
}
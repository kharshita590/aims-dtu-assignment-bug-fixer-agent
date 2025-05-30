# LLM Agent Evaluation and Code Repair

This project evaluates and compares the performance of different language models (LLMs) in repairing Python programs and generating corresponding test cases.

## File Descriptions

- **`agent.py`**  
  Contains the implementation of the LLM agent. Currently, it uses **Gemini 2.0 Flash** as the underlying model.

- **`gemini_repaired_programs/`**  
  This directory contains the code files generated by Gemini for each program in the `python_programs/` directory. Note: Gemini failed to generate valid code for 6 of the programs.

- **`gemini_program_testcases/`**  
  Contains the test cases generated by Gemini for 44 successfully repaired programs.

- **`not_passed_testcases/`**  
  Includes the test cases that Gemini failed to generate. These correspond to the code files for which Gemini did not produce test cases.

- **`repaired_code/`**  
  Contains code files generated by the **Phi-mini Instruct** model. These are included for comparison with Gemini's output.

- **`python_testcases/`**  
  Includes test cases generated for the repaired code from the Phi model.

## Running Tests

To evaluate the generated code using test cases, run the following command:

```bash
pytest gemini_program_testcases/ --correct


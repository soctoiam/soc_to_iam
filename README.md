SOC2IAM

This repository contains a Proof of Concept (PoC) for an end-to-end automated Threat Intelligence pipeline. The system processes unstructured threat reports (text), extracts technical indicators (CPEs & MITRE ATT&CK Techniques) using local LLMs, and automatically applies mitigation rules to a simulated IT infrastructure (CSV assets).

This repository also contains simulation to measure the MTTCA of the TIIR process vs a manual SOC. The used metrics for the simulation and the python file are in the simulation folder.


Quick Start of the TIIR process (Reviewer Guide)

The entire process is encapsulated in a single Kaggle Notebook. You do not need to set up a local environment. The notebook is the folder runtime.

Prerequisites

A Kaggle Account.
Important: Your account must be phone-verified to access GPU resources.

Step-by-Step Execution

1. Get the Files

Download the following files from this repository to your local machine:
 - tiir-process.ipynb (The notebook)

2. Setup Kaggle

Go to Kaggle and click "New Notebook".

File -> Import Notebook: Upload the .ipynb you just downloaded. (Eventually you have to click "Edit" to enter notebook settings)
Configure Hardware: In the right sidebar ("Session options"), set:
Accelerator: GPU T4 x2
Internet: On (Required to fetch scripts & models).

4. Run the Pipeline

Run All Cells.

The first cell (Setup) downloads the model and inference logic from this GitHub repo (~5-8 mins).
The second cell (Run Full Pipeline) executes the demo.

Testing the Scenario

The notebook comes with a pre-configured scenario
You can enter our own vulnerability text, to check the results from the LLM - a match produced by the Loader depends on, wether the CPE is in the test-IAM-data and test-rule-set.

Repository Structure

setup_env.py: The environment bootstrapper. Installs libraries and handles model downloads (handling Kaggle's Model vs Dataset registry).
text2CPE_inference.py: Inference script using a LoRA-finetuned Mistral 7B to extract CPE 2.3 strings.
orchestrator_stix.py: Helper script that bundles inference results into a valid CTI JSON Object.
Loader.py: The logic core. Maps CSVs, checks versions, and applies the mitigation ruleset.
One-Click_Pipeline_Runner.ipynb: The frontend notebook for Kaggle.

Repository Data & Artifact Map

This repository also contains:

text2CPE: Extracts structured CPE components from CVE-style text (vendor/product/target_sw + version ranges) and optionally grounds them deterministically against the official CPE dictionary.

This file structure documents:

- all training-relevant files (external + generated)
- data structure

```text

soc_to_iam
├── simulation
│   ├── Simulation.pdf
│   └── tiir_simulation.py
│
└── tiir_process
    ├── runtime/
    │   ├── notebook for kaggle (.ipynb file)
    │   └── runtimeFiles/
    │       ├── Accounts.CSV
    │       ├── Permissions.CSV
    │       ├── cpe_meta.parquet
    │       ├── cpe_tfidf.npz
    │       ├── vectorizer.pkl
    │       ├── enterprise-attack-v18.1-techniques.xlsx
    │       ├── text2CPE_inference.py
    │       ├── text2technique_inference.py
    │       ├── json_to_cti_parser.py
    │       ├── attac_json_fail.json
    │       ├── attack_json_succ.json
    │       └── setup_env.py
    │   
    └── tiir_transformer/
        └── text2cpe/
            ├── data/
            │   ├── extraction_train_100k_final_clean.jsonl (where we used ~59k for training [GPU constraints]) --> creation script, file to large
            │   └── extraction_eval.jsonl --> creation script, file to large
            ├── artifacts/  # needed for Hybrid Match grounding
            │   ├── cpe_meta.parquet
            │   ├── vectorizer.pkl
            │   └── cpe_tfidf.npz
            └── models/
                └── mistral_CPE_extractor/  # LoRA adapters + tokenizer (loaded in Kaggle)



Base model CPE: mistralai/Mistral-7B-Instruct-v0.3
https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

All available CVE / CPE data: https://nvd.nist.gov/vuln/data-feeds#divJson20Feeds

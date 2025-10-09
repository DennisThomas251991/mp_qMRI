# Frankfurt mp_qMRI (Multi-parametric Quantitative MRI) Pipeline

This repository provides code and Docker environments for post-processing multi-parametric quantitative MRI (mp_qMRI) data, including B1, T1, T2*, H2O, and QSM mapping. The pipeline is designed for vendor-based protocols and supports both Windows and Docker-based workflows.

## Authors

- Dennis C. Thomas & Mariem Ghazouani  
  Neuroradiology Research Group, Co-BIC, University Hospital Frankfurt

## Citation

If you use these codes, please cite:

Dennis C Thomas, Ralf Deichmann, Ulrike Nöth, Christian Langkammer, Mónica Ferreira, Rejane Golbach, Elke Hattingen, Katharina J Wenger,  
"A fast protocol for multi-center, multi-parametric quantitative MRI studies in brain tumor patients using vendor sequences,"  
Neuro-Oncology Advances, 2024;, vdae117, [https://doi.org/10.1093/noajnl/vdae117](https://doi.org/10.1093/noajnl/vdae117)

The QSM mapping code is based on the Graz TGV QSM Algorithm:  
Langkammer, C; Bredies, K; Poser, BA; Barth, M; Reishofer, G; Fan, AP; Bilgic, B; Fazekas, F; Mainero; C; Ropele, S  
"Fast Quantitative Susceptibility Mapping using 3D EPI and Total Generalized Variation."  
Neuroimage. 2015 May 1;111:622-30. [doi:10.1016/j.neuroimage.2015.02.041](https://doi.org/10.1016/j.neuroimage.2015.02.041)

## Repository Structure

- `mp_qMRI_python_codes/`  
  Python scripts for running the pipeline on Windows.

- `mpqMRI_Docker_v1/`  
  Docker-based pipeline for cross-platform reproducibility.

- `mpqMRI_kaapana_v1/`  
  Alternative Docker pipeline with additional integration.

- `Detailed_instructions.txt`  
  Step-by-step user guide for setup and running the pipeline.

- `README.md`  
  This file.

## Getting Started

### Prerequisites

- Docker Desktop: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
- Git: [https://git-scm.com/downloads](https://git-scm.com/downloads)
- Python 3 (if running without Docker)

### Quick Start (Docker)

1. **Clone the repository:**
    ```
    git clone https://github.com/DennisThomas251991/mp_qMRI.git
    cd mp_qMRI/mpqMRI_Docker_v1
    ```

2. **Build the Docker image:**
    ```
    docker build -t qcet .
    ```

3. **Prepare your input data folder**  
   The folder should contain the following files (either `.nii` or `.nii.gz`):
   - M0w_mag
   - T1w_mag
   - M0w_pha
   - T1w_pha
   - B1_1
   - B1_2

4. **Run the pipeline:**
    ```
    docker run -it --name mpqMRI -v "path_to_your_input_folder:/app/input" qcet python3 /app/script_qCET.py --input-folder /app/input
    ```

   For advanced parameter control, use:
    ```
    docker run -it --name mpqMRI -v "path_to_your_input_folder:/app/input" qcet python3 /app/script_user_interaction_test.py --input-folder /app/input
    ```

5. **Outputs**  
   - Final quantitative maps are saved in `Final_qMRI_maps` inside the output folder.
   - Intermediate files are moved to `intermediate_files`.
   - `map_info.txt` describes the output maps.
   - `info.txt` logs the parameters used for processing.

### Running on Windows

See [Detailed_instructions.txt](../Detailed_instructions.txt) and run:

```
python Script_mpqMRI.py
```

Edit paths and parameters in the script to match your data.

## Support

For more details, see the source code or contact the authors.  
GitHub repository: [https://github.com/DennisThomas251991/mp_qMRI](https://github.com/DennisThomas251991/mp_qMRI)

---

**Note:**  
- The pipeline uses FSL, SPM, and the Graz TGV QSM algorithm.  
- See the respective folders and scripts for more details
# Use official Ubuntu base image
FROM ubuntu:20.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 DEBIAN_FRONTEND=noninteractive

# Install base dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        wget bzip2 ca-certificates curl git unzip \
        python3 python3-pip python3-dev cython3\
        libxml2 libxml2-dev libxslt1-dev \
        build-essential dcm2niix xorg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------------------
# Install Miniconda2 (required by TGV-QSM)
# ------------------------------------------
WORKDIR /opt
RUN apt-get update && apt-get install -y wget unzip gcc && \
    wget https://repo.anaconda.com/miniconda/Miniconda2-4.6.14-Linux-x86_64.sh && \
    bash Miniconda2-4.6.14-Linux-x86_64.sh -b -p /opt/miniconda2 && \
    rm Miniconda2-4.6.14-Linux-x86_64.sh

# Add conda to PATH
ENV PATH="/opt/miniconda2/bin:${PATH}"

# ------------------------------------------
# Create environment and install dependencies
# ------------------------------------------
RUN conda install -y cython==0.25.2 numpy pyparsing
RUN pip install scipy==0.17.1 nibabel==2.1.0

# ------------------------------------------
# Install TGV-QSM exactly as authors intended
# ------------------------------------------
WORKDIR /app
COPY TGVQSM-master /app/TGVQSM-master

WORKDIR /app/TGVQSM-master
RUN python setup.py install

# ------------------------------------------
# Test
# ------------------------------------------
WORKDIR /app/TGVQSM-master/test_data
RUN tgv_qsm -p epi3d_test_phase.nii.gz -m epi3d_test_mask.nii.gz -f 2.89 -t 0.027 -o epi3d_test_QSM
# Back to /app
WORKDIR /app

# ------------------------------------------------------------
# FSL installation
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y wget

COPY fslinstaller.py /app
ENV FSLDIR="/usr/local/fsl"
ENV PATH="${FSLDIR}/bin:${PATH}"
RUN python3 /app/fslinstaller.py -d ${FSLDIR}
RUN echo ". ${FSLDIR}/etc/fslconf/fsl.sh" >> ~/.bashrc && \
    . ${FSLDIR}/etc/fslconf/fsl.sh
ENV FSLOUTPUTTYPE=NIFTI_GZ

# Verify FSL
COPY Test_data.nii.gz /app
RUN fslhd /app/Test_data.nii.gz

# ------------------------------------------------------------
# MATLAB Runtime and SPM
# ------------------------------------------------------------
ENV MATLAB_VERSION=R2019b
ENV MCR_VERSION=v97
ENV MCR_UPDATE=9

RUN mkdir /opt/mcr_install /opt/mcr && \
    wget --progress=bar:force -P /opt/mcr_install \
        https://ssd.mathworks.com/supportfiles/downloads/${MATLAB_VERSION}/Release/${MCR_UPDATE}/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_${MATLAB_VERSION}_Update_${MCR_UPDATE}_glnxa64.zip && \
    unzip -q /opt/mcr_install/MATLAB_Runtime_${MATLAB_VERSION}_Update_${MCR_UPDATE}_glnxa64.zip -d /opt/mcr_install && \
    /opt/mcr_install/install -destinationFolder /opt/mcr -agreeToLicense yes -mode silent && \
    rm -rf /opt/mcr_install /tmp/*

ENV SPM_VERSION=12
ENV SPM_REVISION=r7771
ENV LD_LIBRARY_PATH=/opt/mcr/${MCR_VERSION}/runtime/glnxa64:/opt/mcr/${MCR_VERSION}/bin/glnxa64:/opt/mcr/${MCR_VERSION}/sys/os/glnxa64:/opt/mcr/${MCR_VERSION}/sys/opengl/lib/glnxa64:/opt/mcr/${MCR_VERSION}/extern/bin/glnxa64
ENV MCR_INHIBIT_CTF_LOCK=1
ENV SPM_HTML_BROWSER=0

RUN wget --no-check-certificate --progress=bar:force -P /opt \
    https://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/spm12/spm${SPM_VERSION}_${SPM_REVISION}_Linux_${MATLAB_VERSION}.zip && \
    unzip -q /opt/spm${SPM_VERSION}_${SPM_REVISION}_Linux_${MATLAB_VERSION}.zip -d /opt && \
    rm -f /opt/spm${SPM_VERSION}_${SPM_REVISION}_Linux_${MATLAB_VERSION}.zip && \
    /opt/spm${SPM_VERSION}/spm${SPM_VERSION} function exit && \
    chmod +x /opt/spm${SPM_VERSION}/spm${SPM_VERSION}

# ------------------------------------------------------------
# Final setup
# ------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir nibabel nipype scipy scikit-image

WORKDIR /app
COPY . /app

ENTRYPOINT ["python3", "script_run.py"]

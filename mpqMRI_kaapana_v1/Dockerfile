FROM local-only/base-python-cpu:latest

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/opt/conda/bin:$PATH
ENV DEBIAN_FRONTEND="noninteractive"

# Install required packages
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install Anaconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda2-4.6.14-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
# Install additional packages
RUN /opt/conda/bin/conda install -y cython numpy pyparsing && \
    /opt/conda/bin/pip install scipy==0.17.1 nibabel==2.1.0

# Set the working directory
WORKDIR /app

# Assuming your working directory contains the TGVQSM-master-011045626121baa8bfdd6633929974c732ae35e3 folder
COPY TGVQSM-master-011045626121baa8bfdd6633929974c732ae35e3 /app/TGVQSM-master-011045626121baa8bfdd6633929974c732ae35e3

WORKDIR /app/TGVQSM-master-011045626121baa8bfdd6633929974c732ae35e3/

RUN python setup.py install && \
    cd test_data && \
    tgv_qsm  -p epi3d_test_phase.nii.gz -m epi3d_test_mask.nii.gz -f 2.89 -t 0.027 -o epi3d_test_QSM

# Set environment variables for FSL
ENV FSLDIR="/usr/local/fsl"
ENV PATH="${FSLDIR}/bin:${PATH}"

# Install FSL dependencies
RUN apt-get update && \
    apt-get install -y wget

# Copy fslinstaller.py into the Docker image
COPY fslinstaller.py /app

# Run FSL installer script
RUN python /app/fslinstaller.py -d ${FSLDIR} 

# Verify FSL installation
# Echo the FSL configuration command into ~/.bashrc and source it
RUN echo ". ${FSLDIR}/etc/fslconf/fsl.sh" >> ~/.bashrc && \
    . ${FSLDIR}/etc/fslconf/fsl.sh
ENV FSLOUTPUTTYPE=NIFTI_GZ

# Copy the file mGRE_FA_5__24.nii.gz into the Docker image
COPY mGRE_FA_5__24.nii.gz /app

# Verify FSL installation by checking the header information of the copied NIFTI file
RUN fslhd /app/mGRE_FA_5__24.nii.gz


# Update package lists and install Python 3 and pip
RUN apt-get update --fix-missing && \
    apt-get install -y python3 python3-pip

# Create a new layer for installing Python packages
RUN pip3 install --no-cache-dir nibabel nipype scipy scikit-image

# Installing MCR for SPM:
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install \
     unzip xorg wget \
 && apt-get clean \
 && rm -rf \
     /tmp/hsperfdata* \
     /var/*/apt/*/partial \
     /var/lib/apt/lists/* \
     /var/log/apt/term*

# Install MATLAB MCR in /opt/mcr/
ENV MATLAB_VERSION R2019b
ENV MCR_VERSION v97
ENV MCR_UPDATE 9
RUN mkdir /opt/mcr_install \
 && mkdir /opt/mcr \
 && wget --progress=bar:force -P /opt/mcr_install https://ssd.mathworks.com/supportfiles/downloads/${MATLAB_VERSION}/Release/${MCR_UPDATE}/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_${MATLAB_VERSION}_Update_${MCR_UPDATE}_glnxa64.zip \
 && unzip -q /opt/mcr_install/MATLAB_Runtime_${MATLAB_VERSION}_Update_${MCR_UPDATE}_glnxa64.zip -d /opt/mcr_install \
 && /opt/mcr_install/install -destinationFolder /opt/mcr -agreeToLicense yes -mode silent \
 && rm -rf /opt/mcr_install /tmp/*

# Install SPM Standalone in /opt/spm12/
ENV SPM_VERSION 12
ENV SPM_REVISION r7771
ENV LD_LIBRARY_PATH /opt/mcr/${MCR_VERSION}/runtime/glnxa64:/opt/mcr/${MCR_VERSION}/bin/glnxa64:/opt/mcr/${MCR_VERSION}/sys/os/glnxa64:/opt/mcr/${MCR_VERSION}/sys/opengl/lib/glnxa64:/opt/mcr/${MCR_VERSION}/extern/bin/glnxa64
ENV MCR_INHIBIT_CTF_LOCK 1
ENV SPM_HTML_BROWSER 0
# Running SPM once with "function exit" tests the succesfull installation *and*
# extracts the ctf archive which is necessary if singularity is going to be
# used later on, because singularity containers are read-only.
# Also, set +x on the entrypoint for non-root container invocations
RUN wget --no-check-certificate --progress=bar:force -P /opt https://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/spm12/spm${SPM_VERSION}_${SPM_REVISION}_Linux_${MATLAB_VERSION}.zip \
 && unzip -q /opt/spm${SPM_VERSION}_${SPM_REVISION}_Linux_${MATLAB_VERSION}.zip -d /opt \
 && rm -f /opt/spm${SPM_VERSION}_${SPM_REVISION}_Linux_${MATLAB_VERSION}.zip \
 && /opt/spm${SPM_VERSION}/spm${SPM_VERSION} function exit \
 && chmod +x /opt/spm${SPM_VERSION}/spm${SPM_VERSION}

# Configure entry point
WORKDIR /app
COPY . /app
# Unset the DEBIAN_FRONTEND environment variable
ENV DEBIAN_FRONTEND=
ENTRYPOINT ["python", "script_run.py"]


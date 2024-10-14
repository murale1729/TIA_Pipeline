#!/usr/bin/env nextflow

// Define parameters
params.input = "/home/ubuntu/bala/bala/ImpartLabs/tmp/input/sample_small.svs"
params.outdir = "/home/ubuntu/bala/bala/ImpartLabs/tmp/results/"
params.scripts = "/home/ubuntu/bala/bala/ImpartLabs/TIA_Pipeline/single/Scripts"

// Create the output directory if it doesn't exist
new File(params.outdir).mkdirs()

// Define the input channel for the WSI file
Channel.fromPath(params.input).set { wsi_file }

// Process: read_wsi
process read_wsi {
    conda '/path/to/conda/envs/image-processing'
    input:
        path wsi_file

    output:
        path "thumbnail.png"

    publishDir "${params.outdir}", mode: 'copy'

    script:
    """
    python ${params.scripts}/read_wsi.py --input $wsi_file --output thumbnail.png
    """
}

// Process: stain_normalization
process stain_normalization {
    conda '/path/to/conda/envs/image-processing'
    input:
        path wsi_file

    output:
        path "normalized_wsi.png"

    publishDir "${params.outdir}", mode: 'copy'

    echo true

    script:
    """
    python ${params.scripts}/stain_normalization.py --input $wsi_file --output ${params.outdir}normalized_wsi.png --method vahadane
    """
}

// Process: tissue_mask
process tissue_mask {
    conda '/path/to/conda/envs/image-processing'
    input:
        path normalized_wsi

    output:
        path "tissue_mask.png"

    publishDir "${params.outdir}", mode: 'copy'

    script:
    """
    python ${params.scripts}/tissue_mask.py --input $normalized_wsi --output tissue_mask.png --resolution 1.25
    """
}

// Process: nuclei_segmentation
process nuclei_segmentation {
    conda '/path/to/conda/envs/image-processing'
    input:
        path normalized_wsi
        path tissue_mask

    output:
        path "nuclei_result.pkl"

    publishDir "${params.outdir}", mode: 'copy'

    script:
    """
    python ${params.scripts}/nuclei_segmentation.py --input $normalized_wsi --mask $tissue_mask --output nuclei_result.pkl
    """
}



// Workflow Definition
workflow {
    // Start with the WSI file
    def wsi_thumbnail = read_wsi(wsi_file)

    // Perform stain normalization on the original WSI file
    def norm_wsi = stain_normalization(wsi_file)

    // Create tissue mask using normalized WSI
    def tissue_mask_im = tissue_mask(norm_wsi)

    // Perform nuclei segmentation
    def nuclei_res = nuclei_segmentation(
        norm_wsi,
        tissue_mask_im
    )


}

import knime.extension as knext

main_category = knext.category(
    path="/community/",
    level_id="unibz",
    name="UniBZ",
    description="Category for Nodes by the unibz KNIME dev team",
    icon="icons/unibz_icon64.png",
)


 
from knimeVisNodes import imageLoader,Denoising,EdgeDetection,Equalization,KnimeYOLO,SAM,DiceScore,InteractiveView,promptSegSAM,KSurferSSHConnector, KSurferSubjectValidator, KSurferNode, MI_node, icv_normalization, lme_learner_node, lme_apply_node, Hippo_segmentation, Hippo_views, morphometric_measures, subcortical_measures


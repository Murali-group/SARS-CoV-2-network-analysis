library('clusterProfiler')
library('AnnotationHub')
library('org.Hs.eg.db')

# This script will do compute go enrichment on three GO ontology i.e. BP, CC, MF on each of the prediction files on top k predictions.

k=200 #change it to command line argument
predicted_output_path="/media/tassnina/Study/VT/Research_Group/SARSCOV2/SARS-CoV-2-network-analysis/outputs/networks/tissuenet-v2"

prediction_files <- list.files(predicted_output_path, pattern="filtered_pred_scores.csv", full.names=TRUE, recursive=TRUE)
prediction_files

for (file in prediction_files){

	predicted_protein_info = read.table(file, sep = '\t')

	prediction_scores <- predicted_protein_info[,3]

	names(prediction_scores) <- predicted_protein_info[,2]

	prediction_scores <- sort(prediction_scores, decreasing = TRUE)

	top_k_proteins <- names(prediction_scores)[1:k]

	#####BIOLOGICAL PROCESS ENRICHMENT

	ego_BP <- enrichGO(gene          = top_k_proteins,
			universe      = names(prediction_scores),
		      	keyType       = 'UNIPROT',
		        OrgDb         = org.Hs.eg.db,
		        ont           = "BP",
		        pAdjustMethod = "BH",
		        pvalueCutoff  = 0.01,
		        qvalueCutoff  = 0.05)
	
	output_file = paste(dirname(file), 'enrichGO_BP.csv',sep='/')
	
	write.csv(ego_BP,output_file )
	
		
	##### CELLULAR COMPONENT ENRICHMENT
	ego_CC <- enrichGO(gene          = top_k_proteins,
			universe      = names(prediction_scores),
		      	keyType       = 'UNIPROT',
		        OrgDb         = org.Hs.eg.db,
		        ont           = "CC",
		        pAdjustMethod = "BH",
		        pvalueCutoff  = 0.01,
		        qvalueCutoff  = 0.05)
	
	output_file = paste(dirname(file), 'enrichGO_CC.csv',sep='/')
	
	write.csv(ego_CC,output_file )

	
	##### MOLECULAR FUNCTION ENRICHMENT
	ego_MF <- enrichGO(gene          = top_k_proteins,
			universe      = names(prediction_scores),
		      	keyType       = 'UNIPROT',
		        OrgDb         = org.Hs.eg.db,
		        ont           = "MF",
		        pAdjustMethod = "BH",
		        pvalueCutoff  = 0.01,
		        qvalueCutoff  = 0.05)
	
	output_file = paste(dirname(file), 'enrichGO_MF.csv',sep='/')
	
	write.csv(ego_MF,output_file )

}


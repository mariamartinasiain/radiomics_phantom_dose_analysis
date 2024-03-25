import os

# Spécifiez le chemin du dossier où vous voulez modifier les noms de fichiers
dossier = "../DICOM/E2"

# Parcourez tous les fichiers du dossier spécifié
for dossier, sous_dossiers, fichiers in os.walk(dossier):
    for fichier in fichiers:
        chemin_complet = os.path.join(dossier, fichier)
        
        # Vérifiez si le fichier n'a pas d'extension
        if os.path.isfile(chemin_complet) and '.' not in fichier:
            nouveau_nom = chemin_complet + '.dcm'
            
            # Renommez le fichier
            os.rename(chemin_complet, nouveau_nom)
            print(f"Renommé: {fichier} en {nouveau_nom}")

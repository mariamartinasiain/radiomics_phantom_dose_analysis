from harmonization.swin_contrastive.train import Train

class TrainCNN(Train):
    def train_step(self,batch):
        self.optimizer.zero_grad()
        imgs_s = batch["image"].cuda().double()
        print("imgs_s size 1",imgs_s.size())
        if len(imgs_s.size()) == 5:
            imgs_s = imgs_s.view(imgs_s.shape[0] * imgs_s.shape[1],1, imgs_s.shape[2], imgs_s.shape[3], imgs_s.shape[4])
        else :
            imgs_s = imgs_s
        print("imgs_s size 2",imgs_s.size())
    
        # encoder inference
        all_labels = batch["roi_label"].cuda()
        ids = all_labels
        print("ids size 2",ids.size())
        scanner_labels = batch["scanner_label"].cuda()

        latents = self.model(imgs_s)
        print("latents shape",latents.size())
        
        nlatents = [0,0,0,0,0]
        nlatents[4] = latents
        self.contrastive_step(nlatents,ids,latentsize = self.contrastive_latentsize)
        accu = 0
        self.losses_dict['classification_loss'] = 0.0
    

        self.losses_dict['total_loss'] = \
        self.losses_dict['contrast_loss']
            
            
        self.losses_dict['total_loss'].backward()
        self.optimizer.step()
        return self.losses_dict, accu
        
        return compute_accuracy(logits, labels, acc_metric=self.acc_metric)
    def contrastive_step(self, latents,ids,latentsize = 768): #actuellement la loss contrastive est aussi calculé entre sous patchs de la même image, on voudrait eviter ça idealement
        total_num_elements = latents[4].shape[0]
        all_embeddings = torch.empty(total_num_elements, latentsize)
        all_labels = torch.empty(total_num_elements, dtype=torch.long)
        
        offset = 0
        start_idx = 0
        for id in torch.unique(ids):
            #print("id",id)
            boolids = (ids == id)
            btneck = latents[4]  # (batch_size, latentsize, D, H, W)
            btneck = btneck[boolids]
            num_elements = 1
            embeddings = btneck
            contrast_ind = torch.full((num_elements,), offset) #negatives only between different r
            labels = contrast_ind.repeat(btneck.shape[0])            
            end_idx = start_idx + embeddings.shape[0]
            all_embeddings[start_idx:end_idx, :] = embeddings
            all_labels[start_idx:end_idx] = labels
            start_idx = end_idx
            
            offset += num_elements
        
        llss = (self.contrast_loss(all_embeddings, all_labels))
        self.losses_dict['contrast_loss'] = llss

def main():
    from sklearn.preprocessing import LabelEncoder
    device = get_device()
    labels = ['normal1', 'normal2', 'cyst1', 'cyst2', 'hemangioma', 'metastatsis']
    scanner_labels = ['A1', 'A2', 'B1', 'B2', 'C1', 'D1', 'E1', 'E2', 'F1', 'G1', 'G2', 'H1', 'H2']
    encoder = LabelEncoder()
    encoder.fit(labels)
    scanner_encoder = LabelEncoder()
    scanner_encoder.fit(scanner_labels)
    transforms = Compose([
        #PrintDebug(),
        #Resized(keys=["image"],spatial_size = (512,512,343)),
        LoadImaged(keys=["image"]),
        #LazyPatchLoader(roi_size=[64, 64, 32]),
        #DebugTransform2(),
        #EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        EnsureTyped(keys=["image"], device=device, track_meta=False),
        EncodeLabels(encoder=encoder),
        ExtractScannerLabel(),
        EncodeLabels(encoder=scanner_encoder, key='scanner_label'),
        #DebugTransform(),
        #DebugTransform2(),
        
    ])
    #jsonpath = "./registered_light_dataset_info_10.json"
    jsonpath = "./dataset_info_cropped.json"
    
    data_list = load_data(jsonpath)
    train_data, test_data = create_datasets(data_list,test_size=0.00)
    #model = get_model(target_size=(64, 64, 32))
    model = get_oscar_for_training()
    #transfering model on device 
    model.to(device)
    
    train_dataset = SmartCacheDataset(data=train_data, transform=transforms,cache_rate=1,progress=True,num_init_workers=8, num_replace_workers=8,replace_rate=0.1)
    test_dataset = SmartCacheDataset(data=test_data, transform=transforms,cache_rate=0.1,progress=True,num_init_workers=8, num_replace_workers=8)

    train_loader = ThreadDataLoader(train_dataset, batch_size=8, shuffle=True,collate_fn=custom_collate_fn)
    train_loader = ThreadDataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=custom_collate_fn)
    test_loader = ThreadDataLoader(test_dataset, batch_size=3, shuffle=False,collate_fn=custom_collate_fn)

    data_loader = {'train': train_loader, 'test': test_loader}
    dataset = {'train': train_dataset, 'test': test_dataset}
    
    
    
    print(f"Le nombre total de poids dans le modèle est : {count_parameters(model)}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.005) 
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    trainer = TrainCNN(model, data_loader, optimizer, lr_scheduler, 100,dataset,contrastive_latentsize=2048,savename="contrast_oscar.pth",ortho_reg=0.001)
    trainer.train()
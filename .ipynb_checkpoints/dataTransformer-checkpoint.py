class DataTransformer():
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image = augmented['image']

        return image, label
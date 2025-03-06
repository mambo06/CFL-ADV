Implementation of paper "Contrastive Federated Learning with Tabular Data Silos"


To run any specific dataset use:
```
python train.py -d datasetName -e epochNumber {args}
```
Arguments available
```
"-d", "--dataset", type=str, default="mnist", help='Name of the dataset to use. It should have a config file with the same name.'
"-g", "--gpu", dest='gpu', action='store_true',  help='Used to assign GPU as the device, assuming that GPU is available'
"-m", "--mps", dest='mps', action='store_true',  help='Used to assign MAC M1 GPU as the device, assuming that GPU is available'
"-ng", "--no_gpu", dest='gpu', action='store_false', help='Used to assign CPU as the device'
"-dn", "--device_number", type=str, default='0', help='Defines which GPU to use. It is 0 by default'
"-lc", "--local", dest='local', action='store_true, help='Non federatede learning'
"-e", "--epoch", type=int, default=5, help='epoch'
"-c", "--client", type=int, default=4, help='number of client > 1'
"-cd", "--clientdrop", type=float, default=0, help='percentage of client drop from number of client'
"-dd", "--datadrop", type=float, default=0, help='percentage of data from in clientdrop'
"-ci", "--classimbalance", type=float, default=0, help='percentage of class drop from clientdrop'
```
Data Can be downloaded from
```

@misc{misc_tuandromd_(tezpur_university_android_malware_dataset)_855,
  author       = {Borah,Parthajit and Bhattacharyya,Dhruba K.},
  title        = {{TUANDROMD (Tezpur University Android Malware Dataset)}},
  year         = {2023},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5560H}
}
@misc{misc_covertype_31,
  author       = {Blackard,Jock},
  title        = {{Covertype}},
  year         = {1998},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C50K5N}
}
@misc{misc_dataset_for_sensorless_drive_diagnosis_325,
  author       = {Bator,Martyna},
  title        = {{Dataset for Sensorless Drive Diagnosis}},
  year         = {2015},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5VP5F}
}

@misc{misc_adult_2,
  author       = {Becker,Barry and Kohavi,Ronny},
  title        = {{Adult}},
  year         = {1996},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5XW20}
}
@misc{misc_blogfeedback_304,
  author       = {Buza,Krisztian},
  title        = {{BlogFeedback}},
  year         = {2014},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C58S3F}
}

```

This code was modified from 

SubTab By Talip Ucar

echo 'Downloading weights of pretrained models ...'
gshell cd --with-id 1Yc-xzYw3yEJE3_64KgF2sfFdw_u8XZAc #download weights of the model
gshell download transformations.pk 
gshell download tvsum_random_non_overlap_0.6271.tar.pth
gshell download summe_random_non_overlap_0.5359.tar.pth
gshell download flow_imagenet.pt
gshell download rgb_imagenet.pt
gshell download r3d101_KM_200ep.pth
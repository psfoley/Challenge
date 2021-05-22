def brats_labels(output, 
                 target, 
                 class_list== ['4', '1||4', '1||2||4'], 
                 binarized=True, 
                 **kwargs):
    # take output and target and create: (output_<task>, lable_<task>)
    # for tasks in ['enhancing', 'core', 'whole']
    # these can be binary (per-voxel) decisions (if binarized==True) or float valued
    if binarized:
        output_enhancing = binarize_output(output=output, 
                                           class_list=class_list, 
                                           modality='ET')
        
        output_core = binarize_output(output=output, 
                                      class_list=class_list, 
                                      modality='TC')
        
        output_whole = binarize_output(output=output, 
                                       class_list=class_list, 
                                       modality='WT')
       
    # We detect specific use_cases here, and force a change in the code when another is wanted.
    # In all cases, we rely on the order of class_list !!!
    if list(class_list) == [0, 1, 2, 4]:
        if not binarized:
            # signal is channel 3 based on known class_list
            output_enhancing = output[:,3,:,:,:]

            # core signal comes from channels 1 or 3 based on known class_list
            output_channels_1_3 = torch.cat([output[:,1:2,:,:,:], output[:,3:4,:,:,:]], dim=1)
            output_core = torch.max(output_channels_1_3,dim=1).values

            # whole signal comes from channels 1, 2, or 3 based on known class_list
            output_whole = torch.max(output[:,1:,:,:,:],dim=1).values
        
        # signal is channel 3 based on known class_list
        target_enhancing = target[:,3,:,:,:]

        # core signal comes from channels 1 or 3 based on known class_list
        target_channels_1_3 = torch.cat([target[:,1:2,:,:,:], target[:,3:4,:,:,:]],dim=1)
        target_core = torch.max(target_channels_1_3,dim=1).values
    
        # whole signal comes from channels 1, 2, or 3 based on known class_list
        target_whole = torch.max(target[:,1:,:,:,:],dim=1).values
    
    elif list(class_list) == ['4', '1||4', '1||2||4']:
        # In this case we track only enhancing tumor, tumor core, and whole tumor (no background class).
    
        if not binarized:

            # enhancing signal is channel 0 because of known class_list with fused labels
            output_enhancing = output[:,0,:,:,:]

            # core signal is channel 1 because of known class_list with fused labels
            output_core = output[:,1,:,:,:]

            # whole signal is channel 2 because of known class_list with fused labels
            output_whole = output[:,2,:,:,:]
        
        
        # enhancing signal is channel 0 because of known class_list with fused labels
        target_enhancing = target[:,0,:,:,:]
        
        # core signal is channel 1 because of known class_list with fused labels
        target_core = target[:,1,:,:,:]
        
        # whole signal is channel 2 because of known class_list with fused labels
        target_whole = target[:,2,:,:,:]
    else:
        raise ValueError('No implementation for this model class_list: ', class_list)

    check_shapes_same(output=output_enhancing, target=target_enhancing)
    check_shapes_same(output=output_core, target=target_core)
    check_shapes_same(output=output_whole, target=target_whole)

    return {'outputs': {'ET': output_enhancing, 
                        'TC': output_core,
                        'WT': output_whole},
            'targets': {'ET': target_enhancing, 
                        'TC': target_core, 
                        'WT': target_whole}}


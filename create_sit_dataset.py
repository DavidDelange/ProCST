def main(opt):
    model = torch.load(opt.trained_procst_path)
    for i in range(len(model)):
        model[i].eval()
        model[i] = torch.nn.DataParallel(model[i])
        model[i].to(opt.device)
    opt.num_scales = opt.curr_scale = len(model)-1
    source_train_loader = CreateSrcDataLoader(opt, get_image_label=True, get_sit_image=True)
    if opt.skip_created_files:
        already_created = next(os.walk(opt.sit_output_path))[2]
        for f in already_created:
            if f in source_train_loader.dataset.img_ids:
                source_train_loader.dataset.img_ids.remove(f)
    print('Number of images to convert: %d' % len(source_train_loader.dataset.img_ids))
    i = 0
    for source_scales, filenames, labels, backgrounds in tqdm(source_train_loader):
        if i>500:
            break
        for i in range(len(source_scales)):
            source_scales[i] = source_scales[i].to(opt.device)
        sit_batch = concat_pyramid_eval(model, source_scales, opt)
        for i, filename in enumerate(filenames):
            save_image(norm_image(sit_batch[i]), os.path.join(opt.sit_output_path, 'samples_styled', filename+'.png'))
            save_image(norm_image(source_scales[len(source_scales)-1]), os.path.join(opt.sit_output_path, 'samples', filename+'.png'))
            label_tensor = torch.tensor(colorize_mask(labels[i]), dtype=torch.float32)
            save_image(label_tensor, os.path.join(opt.sit_output_path, 'targets_perfect', filename+'.png'))
            save_image(backgrounds[i], os.path.join(opt.sit_output_path, 'targets_road_with_gaps', filename+'.png')) 
            i+=1
    print('Finished Creating SIT Dataset.')


if __name__ == "__main__":
    from core.config import get_arguments, post_config
    parser = get_arguments()
    opt = parser.parse_args()
    opt.source = 'synth_daformer'
    opt.src_data_dir = '/raw_synth_gsx_008/raw-synth-gsx-008'
    opt = post_config(opt)
    from tqdm import tqdm
    from data_handlers import CreateSrcDataLoader
    import torch
    from core.config import get_arguments, post_config
    from core.functions import norm_image, colorize_mask
    from core.training import concat_pyramid_eval
    import os
    from torchvision.utils import save_image
    main(opt)


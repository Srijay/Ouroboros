import argparse
import math
from collections import defaultdict
import json
import csv
from torchvision.utils import save_image
import configparser
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from scipy.stats import pearsonr,spearmanr,kendalltau
from sklearn.metrics import r2_score
import torchvision.utils as vutils
from data.data import HyperionDataset
from model import GenerativeModel
from discriminators import Pix2PixDiscriminator, PatchDiscriminator, BigGanDiscriminator
from generators import weights_init
from losses import get_gan_losses
from utils import *
import os
import gradio as gr
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PIL import Image
from combat.pycombat import pycombat

parser = argparse.ArgumentParser()

parser.add_argument('--image_dir',
                    default=r'F:\Datasets\RA\SpecTX\Srijay\patches\patches_256')

# Optimization hyperparameters
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_iterations', default=800000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)

# Switch the generator to eval mode after this many iterations
parser.add_argument('--eval_mode_after', default=9000000, type=int)

# Dataset options
parser.add_argument('--num_train_samples', default=10, type=int)
parser.add_argument('--num_val_samples', default=10, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=0, type=int)

# Image Generator options
parser.add_argument('--generator', default='biggan')  # dcgan or pix2pix or residual or biggan
parser.add_argument('--l1_pixel_image_loss_weight', default=100.0, type=float)  # 1.0
parser.add_argument('--normalization', default='instance')
parser.add_argument('--activation', default='leakyrelu-0.2')

# Generic discriminator options
parser.add_argument('--discriminator_loss_weight', default=1, type=float)  # 0.01
parser.add_argument('--gan_loss_type', default='gan')

# Image discriminator
parser.add_argument('--discriminator', default='biggan')  # patchgan or standard or biggan

# Output options
parser.add_argument('--print_every', default=100, type=int)
parser.add_argument('--timing', default=False, type=bool_flag)
parser.add_argument('--checkpoint_every', default=1000, type=int)

# Experiment related parameters
parser.add_argument('--experimentname', default='hyperion_generation_fromsinglevector_biggan_3_A')
parser.add_argument('--output_dir', default=os.path.join('./output'))
parser.add_argument('--checkpoint_name', default='model.pt')

parser.add_argument('--checkpoint_path', default='./output/hyperion_generation_fromsinglevector_biggan_3_A/model/model.pt')
parser.add_argument('--restore_from_checkpoint', default=True, type=bool_flag)
parser.add_argument('--test_output_dir', default=os.path.join(r'F:\Datasets\RA\SpecTX\Srijay\results\generation\hyperion_generation_fromsinglevector_biggan_3_A'))

# If you want to test model, set mode to test, or gradio
parser.add_argument('--mode', default='train', type=str)


def ZSSTransform(z):
    zlog = np.log10(z[z>0])
    tzlog = (zlog-np.mean(zlog))/np.std(zlog)
    tzlog_min = np.min(tzlog)-1
    tz = z.copy()
    tz[tz>0] = tzlog
    tz[tz==0] = tzlog_min
    return tz


def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    curr_loss = curr_loss * weight
    loss_dict[loss_name] = curr_loss.item()
    if total_loss is not None:
        total_loss += curr_loss
    else:
        total_loss = curr_loss
    return total_loss


def build_dsets(args):

    dset_kwargs = {
        'image_dir': args.image_dir,
        'mode': args.mode
    }

    dset = HyperionDataset(**dset_kwargs)

    num_imgs = len(dset)
    print(args.mode + ' dataset has %d images' % (num_imgs))

    return dset


def build_loader(args):

    dset = build_dsets(args)

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': True,
    }

    loader = DataLoader(dset, **loader_kwargs)

    return loader


def build_model(args):
    kwargs = {
        'normalization': args.normalization,
        'activation': args.activation,
        'mode': args.mode,
        'generator_name': args.generator
    }
    model = GenerativeModel(**kwargs)
    return model, kwargs


def build_img_discriminator(args):

    if (args.discriminator == 'patchgan'):
        discriminator = Pix2PixDiscriminator(in_channels=3)
    elif (args.discriminator == 'standard'):
        d_kwargs = {
            'arch': args.d_img_arch,
            'normalization': args.d_normalization,
            'activation': args.d_activation,
            'padding': args.d_padding,
        }
        discriminator = PatchDiscriminator(**d_kwargs)
    elif (args.discriminator == 'biggan'):
        discriminator = BigGanDiscriminator()
    else:
        raise "Give proper name of discriminator"

    discriminator = discriminator.apply(weights_init)

    return discriminator


def check_model(args, t, loader, model, mode):

    experiment_output_dir = os.path.join(args.output_dir, args.experimentname)

    num_samples = 0

    output_dir = os.path.join(experiment_output_dir, "training_output", mode)
    mkdir(output_dir)
    # model.eval()
    n_samples = 0

    with torch.no_grad():
        for batch in loader:

            img_name, pooled_features, image_gt = batch

            if torch.cuda.is_available():
                pooled_features = pooled_features.cuda()
                image_gt = image_gt.cuda()

            # pooled_features = pooled_features.reshape([args.batch_size, pooled_features.shape[1], 1, 1])

            image_pred = model(hyperion_features=pooled_features.float())

            image_pred = image_pred.cuda()
            image_gt = image_gt

            n_samples += 1
            if n_samples >= 5:
                break

            im_initial = img_name[0].split(".")[0]

            image_gt_path = os.path.join(output_dir, im_initial + "_gt_image.png")
            save_image(image_gt, image_gt_path)

            if (image_pred is not None):

                image_pred_path = os.path.join(output_dir, im_initial + "_pred_image.png")
                save_image(image_pred, image_pred_path)


def compute_metrics(total_real_counts, total_predicted_counts):
    #Compute pearson, spearman, kendalltau and r2
    metric_dict = {}
    i=0
    columns_to_count = "SMAa,CD11b,CD44,CD31,CDK4,YKL40,CD11c,HIF1a,CD24,TMEM119,OLIG2,GFAP,VISTA,IBA1,CD206,PTEN,NESTIN,TCIRG1,CD74,MET,P2RY12,CD163,S100B,cMYC,pERK,EGFR,SOX2,HLADR,PDGFRa,MCT4,DNA1,DNA3,MHCI,CD68,CD14,KI67,CD16,SOX10"
    columns_to_count = columns_to_count.split(",")
    for type in columns_to_count:
        real_counts = total_real_counts[:,i]
        predicted_counts = total_predicted_counts[:,i]
        metric_dict[type] = [pearsonr(real_counts, predicted_counts)[0],
                             spearmanr(real_counts, predicted_counts)[0],
                             kendalltau(real_counts, predicted_counts)[0],
                             r2_score(real_counts, predicted_counts)]
        i+=1
    return metric_dict


def generate_report(report_name, metric_dict):
    output_file = os.path.join(args.test_output_dir, report_name+".csv")
    csv_file = open(output_file, "w")
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['type', 'pearson', 'spearman', 'kendalltau', 'r2'])
    for type in metric_dict:
        metrics = metric_dict[type]
        metrics = [str(metric) for metric in metrics]
        writer.writerow([type] + metrics)


def test_model(args, loader, model, image_discriminator):

    gt_image_output_dir = os.path.join(args.test_output_dir,"real")
    pred_image_output_dir = os.path.join(args.test_output_dir,"synthetic")
    mkdir(gt_image_output_dir)
    mkdir(pred_image_output_dir)

    real_counts = []
    pred_counts = []

    with torch.no_grad():

        for batch in loader:

            img_name, pooled_features, image_gt = batch

            if torch.cuda.is_available():
                pooled_features = pooled_features.cuda()
                image_gt = image_gt.cuda()

            # print("Here-------------------------------------------------", img_name)
            # print(pooled_features)
            # pooled_features = pooled_features.reshape([args.batch_size, pooled_features.shape[1], 1, 1])
            image_pred = model(hyperion_features=pooled_features.float())
            _, predicted_expression = image_discriminator(image_gt.float(), pooled_features.float())

            real_counts_batch_np = pooled_features.cpu().detach().numpy().tolist()
            pred_counts_batch_np = predicted_expression.cpu().detach().numpy().tolist()

            real_counts = real_counts + real_counts_batch_np
            pred_counts = pred_counts + pred_counts_batch_np

            image_pred = image_pred.cuda()
            image_gt = image_gt

            image_gt_path = os.path.join(gt_image_output_dir, img_name[0])
            save_image(image_gt, image_gt_path)

            image_pred_path = os.path.join(pred_image_output_dir, img_name[0])
            save_image(image_pred, image_pred_path)

        real_counts = np.array(real_counts)
        pred_counts = np.array(pred_counts)
        np.save("A_real.py", real_counts)
        np.save("A_syn.py", pred_counts)

        # print(real_counts.shape)
        # print(pred_counts.shape)
        # exit()

        # generate overall report
        overall_metrics = compute_metrics(real_counts, pred_counts)
        generate_report(args.experimentname + "_overall_report", overall_metrics)


def interpolate(args, model):
    cell_counts_path = r"F:\Datasets\RA\SpecTX\Srijay\protein_expressions_celllevel.csv"
    cell_df = pd.read_csv(cell_counts_path)
    cell_df = cell_df[cell_df['VisSpot'].notna()]
    wsi_image_ids = cell_df.VisSpot.apply(lambda x: str(x)[-2:])  # get the image IDs
    biomarkers = "SMAa,CD11b,CD44,CD31,CDK4,YKL40,CD11c,HIF1a,CD24,TMEM119,OLIG2,GFAP,VISTA,IBA1,CD206,PTEN,NESTIN,TCIRG1,CD74,MET,P2RY12,CD163,S100B,cMYC,pERK,EGFR,SOX2,HLADR,PDGFRa,MCT4,DNA1,DNA3,MHCI,CD68,CD14,KI67,CD16,SOX10"
    biomarkers = biomarkers.split(",")
    cell_df = cell_df[['VisSpot', 'Location_Center_Y', 'Location_Center_X'] + biomarkers]
    cell_df[biomarkers] = cell_df[biomarkers].apply(ZSSTransform)  # apply the log transform
    cell_df[biomarkers] = pycombat(cell_df[biomarkers].T, wsi_image_ids).T  # batch correction
    df_hyperion = cell_df.groupby('VisSpot', as_index=False).mean()

    config = configparser.ConfigParser()
    config.read(r'cell_types')
    parameters = config['celltypes']
    biomarkers_list = parameters['biomarkers'].split(",")

    def generate_image(*biomarkers):
        biomarkers_transformed = torch.Tensor(np.array(biomarkers)).cuda()
        biomarkers_transformed = biomarkers_transformed[None, :]
        predimage = model(biomarkers_transformed)
        return predimage

    VisSpot_start = "GGGTCAGGAGCTAGAT-1-C1" #C1_65x55
    VisSpot_end = "TCTTACGGCATCCGAC-1-C1" #C1_25x29
    featues_start = df_hyperion.loc[df_hyperion['VisSpot'] == VisSpot_start, biomarkers_list].values.flatten().tolist()
    featues_end = df_hyperion.loc[df_hyperion['VisSpot'] == VisSpot_end, biomarkers_list].values.flatten().tolist()

    sorted_indices = np.argsort(featues_start)

    num_interpolations = 10
    interpolated_vectors=[]

    for i in range(num_interpolations):
        ivector = [(num_interpolations - i) * x + i * y for x, y in zip(featues_start, featues_end)]
        ivector = [x / num_interpolations for x in ivector]
        interpolated_vectors.append(ivector)

    tensor_vectors = np.array(interpolated_vectors)
    df = pd.DataFrame(tensor_vectors, columns=biomarkers_list)
    df.to_csv("interpolation_data.csv", index=False)

    min_value = min(np.min(vector) for vector in interpolated_vectors)
    max_value = max(np.max(vector) for vector in interpolated_vectors)

    images = [generate_image(vector) for vector in interpolated_vectors]

    fig, axes = plt.subplots(2, num_interpolations, figsize=(18, 6))
    for i, (interpolated_vector, image) in enumerate(zip(interpolated_vectors, images)):
        axes[1, i].imshow(image.squeeze().permute(1, 2, 0).detach().cpu().numpy())
        dimensions = np.arange(1, 39)
        axes[0, i].bar(dimensions, [interpolated_vector[i] for i in sorted_indices], color=plt.cm.viridis(np.linspace(0, 1, 38)), bottom=min_value)
        save_image(image, str(i+1)+'.png')
        # axes[0, i].set_yticks(np.arange(0, 38, step=1))
        # axes[0, i].set_yticklabels([f'{int(val)}' for val in np.arange(0, 38, step=1)])
        # axes[0, i].set_ylabel('Frequency')
        # for l, p in zip(biomarkers, patches):
        #     axes[0, i].text(p.get_x() + p.get_width() / 2, -1, l, fontsize='xx-small', rotation=70)
        # for c, p in zip(colors, patches):
        #     p.set_facecolor(c)
    plt.tight_layout()
    # Hide the axes
    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')
    plt.show()

    return


def calculate_model_losses(args, image_gt, image_pred):
    total_loss = torch.zeros(1).to(image_gt)
    losses = {}

    # Image L1 Loss
    l1_pixel_loss_images = F.l1_loss(image_pred, image_gt.float())
    total_loss = add_loss(total_loss, l1_pixel_loss_images, losses, 'L1_pixel_loss_images',
                          args.l1_pixel_image_loss_weight)

    # l2_prediction_loss = F.mse_loss(pred_expression, gt_expression.float())
    # total_loss = add_loss(total_loss, l2_prediction_loss, losses, 'prediction_loss',
    #                       args.l1_pixel_image_loss_weight)

    return total_loss, losses


def main(args):

    torch.cuda.empty_cache()

    experiment_output_dir = os.path.join(args.output_dir, args.experimentname)
    model_dir = os.path.join(experiment_output_dir, "model")

    if torch.cuda.is_available():
        float_dtype = torch.cuda.FloatTensor
    else:
        float_dtype = torch.FloatTensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (args.mode == "train"):

        mkdir(experiment_output_dir)
        mkdir(model_dir)

        with open(os.path.join(experiment_output_dir, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    loader = build_loader(args)

    model, model_kwargs = build_model(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5,0.999))

    # Image Discriminator
    image_discriminator = build_img_discriminator(args)
    if image_discriminator is not None:
        image_discriminator.cuda()
        image_discriminator.type(float_dtype)
        image_discriminator.train()
        optimizer_d_image = torch.optim.Adam(image_discriminator.parameters(),
                                             lr=args.learning_rate, betas=(0.5,0.999))

    gan_g_loss, gan_d_loss = get_gan_losses(args.gan_loss_type)

    if args.restore_from_checkpoint or args.mode == "test" or args.mode == "gradio" or args.mode == "interpolate":

        print("Restoring")
        restore_path = args.checkpoint_path

        checkpoint = torch.load(restore_path, map_location="cpu")

        model.load_state_dict(checkpoint['model_state'], strict=True)

        if (args.mode == "train"):
            optimizer.load_state_dict(checkpoint['optim_state']) #strict argument is not supported here

        if image_discriminator is not None:
            image_discriminator.load_state_dict(checkpoint['d_image_state'])
            optimizer_d_image.load_state_dict(checkpoint['d_image_optim_state'])
            image_discriminator.cuda()

        if (args.mode == "test"):
            model.eval()
            test_model(args, loader, model, image_discriminator)
            print("Testing has been done and results are saved")
            return

        if (args.mode == "interpolate"):
            model.eval()
            interpolate(args, model)
            print("Testing has been done and results are saved")
            return

        if (args.mode == "gradio"):

            model.eval()

            cell_counts_path = r"F:\Datasets\RA\SpecTX\Srijay\protein_expressions_celllevel.csv"
            cell_df = pd.read_csv(cell_counts_path)
            cell_df = cell_df[cell_df['VisSpot'].notna()]
            wsi_image_ids = cell_df.VisSpot.apply(lambda x: str(x)[-2:])  # get the image IDs
            biomarkers = "SMAa,CD11b,CD44,CD31,CDK4,YKL40,CD11c,HIF1a,CD24,TMEM119,OLIG2,GFAP,VISTA,IBA1,CD206,PTEN,NESTIN,TCIRG1,CD74,MET,P2RY12,CD163,S100B,cMYC,pERK,EGFR,SOX2,HLADR,PDGFRa,MCT4,DNA1,DNA3,MHCI,CD68,CD14,KI67,CD16,SOX10"
            biomarkers = biomarkers.split(",")
            cell_df = cell_df[['VisSpot', 'Location_Center_Y', 'Location_Center_X'] + biomarkers]
            cell_df[biomarkers] = cell_df[biomarkers].apply(ZSSTransform)  # apply the log transform
            cell_df[biomarkers] = pycombat(cell_df[biomarkers].T, wsi_image_ids).T  # batch correction
            df_hyperion = cell_df.groupby('VisSpot', as_index=False).mean()

            config = configparser.ConfigParser()
            config.read(r'cell_types')
            parameters = config['celltypes']
            biomarkers_list = parameters['biomarkers'].split(",")

            def get_results(*biomarkers):
                biomarkers_transformed = torch.Tensor(np.array(biomarkers)).cuda()
                biomarkers_transformed = biomarkers_transformed[None,:]
                predimage = model(biomarkers_transformed)
                save_image(predimage, "image.png")
                image = Image.open("image.png")
                return image

            VisSpot = "AAACACCAATAACTGC-1-B1"
            default_features = df_hyperion.loc[df_hyperion['VisSpot'] == VisSpot, biomarkers_list].values.flatten().tolist()

            range_sliders = []
            index = 0
            for biomarker in biomarkers_list:
                range_sliders.append(gr.Slider(minimum=df_hyperion[biomarker].min(), maximum=df_hyperion[biomarker].max(), value = default_features[index], label=biomarker))
                index+=1

            demo = gr.Interface(
                get_results,
                inputs = range_sliders,
                outputs=gr.Image(label="Generated Image"),
                title="Tissue Image Generation using Protein Biomarkers"
            )

            demo.launch(share=True)

            return

        t = 0

        epoch = checkpoint['counters']['epoch']

        print("Starting Epoch : ", epoch)

    else:

        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'model_kwargs': model_kwargs,
            'losses_ts': [],
            'losses': defaultdict(list),
            'd_losses': defaultdict(list),
            'checkpoint_ts': [],
            'train_batch_data': [],
            'train_samples': [],
            'train_iou': [],
            'val_batch_data': [],
            'val_samples': [],
            'val_losses': defaultdict(list),
            'val_iou': [],
            'norm_d': [],
            'norm_g': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'model_state': None, 'model_best_state': None, 'optim_state': None,
            'd_obj_state': None, 'd_obj_best_state': None, 'd_obj_optim_state': None,
            'd_img_state': None, 'd_img_best_state': None, 'd_img_optim_state': None,
            'd_mask_state': None, 'best_t': [],
        }

    # Loss Curves
    training_loss_out_dir = os.path.join(experiment_output_dir, 'training_loss_graph')
    mkdir(training_loss_out_dir)

    def draw_curve(epoch_list, loss_list, loss_name):
        plt.clf()
        plt.plot(epoch_list, loss_list, 'bo-', label=loss_name)
        plt.legend()
        plt.savefig(os.path.join(training_loss_out_dir, loss_name + '.png'))

    epoch_list = []
    monitor_epoch_losses = defaultdict(list)

    while True:

        if t >= args.num_iterations:
            break

        for batch in loader:

            if t == args.eval_mode_after:
                print('switching to eval mode')
                model.eval()
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

            img_name, pooled_features, image_gt = batch

            if torch.cuda.is_available():
                pooled_features = pooled_features.cuda()
                image_gt = image_gt.cuda()

            # pooled_features = pooled_features.reshape([args.batch_size, pooled_features.shape[1], 1, 1])
            image_pred = model(hyperion_features=pooled_features.float())
            image_pred = image_pred.cuda()
            # predicted_expression = predicted_expression.cuda()

            total_loss, losses = calculate_model_losses(args, image_gt, image_pred)

            if image_discriminator is not None:
                if(args.discriminator == 'biggan'):
                    scores_image_fake, fake_prediction = image_discriminator(image_pred, pooled_features.float())
                else:
                    scores_image_fake = image_discriminator(image_pred)
                scores_image_fake = scores_image_fake.cuda()
                weight = args.discriminator_loss_weight
                total_loss = add_loss(total_loss, gan_g_loss(scores_image_fake), losses, 'g_gan_image_loss', weight)

                l2_prediction_loss = F.mse_loss(fake_prediction, pooled_features.float())
                total_loss = add_loss(total_loss, l2_prediction_loss, losses, 'fake_g_prediction_loss',
                                      args.l1_pixel_image_loss_weight)

            losses['total_loss'] = total_loss.item()
            if not math.isfinite(losses['total_loss']):
                print('WARNING: Got loss = NaN, not backpropping')
                continue

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            image_fake = image_pred.detach()
            image_real = image_gt.detach()

            if image_discriminator is not None:
                d_image_losses = LossManager()  # For image
                if (args.discriminator == 'biggan'):
                    scores_fake, _ = image_discriminator(image_fake, pooled_features.float())
                    scores_real, real_d_prediction = image_discriminator(image_real.float(), pooled_features.float())
                else:
                    scores_fake = image_discriminator(image_fake)
                    scores_real = image_discriminator(image_real.float())
                d_image_gan_loss = gan_d_loss(scores_real, scores_fake)
                d_image_losses.add_loss(d_image_gan_loss, 'd_image_gan_loss')

                l2_prediction_loss = F.mse_loss(real_d_prediction, pooled_features.float())
                d_image_losses.add_loss(l2_prediction_loss, 'real_d_prediction_loss')

                optimizer_d_image.zero_grad()
                d_image_losses.total_loss.backward()
                optimizer_d_image.step()
                image_discriminator.cuda()

            t += 1

            if t % args.print_every == 0:

                print('t = %d / %d' % (t, args.num_iterations))
                for name, val in losses.items():
                    print(' G [%s]: %.4f' % (name, val))

                if image_discriminator is not None:
                    for name, val in d_image_losses.items():
                        print(' D_img [%s]: %.4f' % (name, val))

            if t % args.checkpoint_every == 0:

                print('checking on train')
                check_model(args, t, loader, model, "train")

                checkpoint['model_state'] = model.state_dict()

                if image_discriminator is not None:
                    checkpoint['d_image_state'] = image_discriminator.state_dict()
                    checkpoint['d_image_optim_state'] = optimizer_d_image.state_dict()

                checkpoint['optim_state'] = optimizer.state_dict()
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint_path = os.path.join(model_dir, args.checkpoint_name)
                print('Saving checkpoint to ', checkpoint_path)
                torch.save(checkpoint, checkpoint_path)

        # Plot the loss curves
        epoch += 1
        epoch_list.append(epoch)
        for k, v in losses.items():
            monitor_epoch_losses[k].append(v)
            draw_curve(epoch_list, monitor_epoch_losses[k], k)


if __name__ == '__main__':
    print("CONTROL")
    args = parser.parse_args()
    main(args)
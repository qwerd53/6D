def debug_loftr(self,batch):
    """
    Debug LoFTR + mask filtering + visualization + top-8 mconf filtering
    保存原图、GT mask过滤图、LoFTR匹配可视化图
    """

    import os
    import torch
    import numpy as np
    from PIL import Image
    import torchvision.utils as vutils
    import matplotlib.pyplot as plt
    import cv2

    save_dir = "debug_loftr"
    os.makedirs(save_dir, exist_ok=True)

    B = batch['image0'].shape[0]
    num_save = min(3, B)
    device = batch['image0'].device

    for i in range(num_save):
        img0 = batch['image0'][i:i+1]  # [1,3,H,W]
        img1 = batch['image1'][i:i+1]
        mask0 = batch['mask0_gt'][i]   # [H,W]
        mask1 = batch['mask1_gt'][i]

        # -------------------------
        # 1) 保存原图
        # -------------------------
        vutils.save_image(img0, f"{save_dir}/image0_raw_{i}.png", normalize=True)
        vutils.save_image(img1, f"{save_dir}/image1_raw_{i}.png", normalize=True)

        # -------------------------
        # 2) GT mask 过滤图
        # -------------------------
        mask0_t = mask0.unsqueeze(0).unsqueeze(0).float()
        mask1_t = mask1.unsqueeze(0).unsqueeze(0).float()

        img0_filtered = img0 * mask0_t
        img1_filtered = img1 * mask1_t

        vutils.save_image(img0_filtered, f"{save_dir}/image0_gtmask_{i}.png", normalize=True)
        vutils.save_image(img1_filtered, f"{save_dir}/image1_gtmask_{i}.png", normalize=True)

        # -------------------------
        # 3）送入 LoFTR
        # -------------------------
        match_batch = {'image0': img0_filtered, 'image1': img1_filtered}
        with torch.no_grad():
            self.matcher.eval()
            self.matcher(match_batch)

        mkpts0 = match_batch['mkpts0_f'].detach().cpu().numpy()  # [N, 2]
        mkpts1 = match_batch['mkpts1_f'].detach().cpu().numpy()  # [N, 2]
        mconf = match_batch['mconf'].detach().cpu().numpy()  # [N]

        #if (self.useGTmask):
            # if使用 GT 掩码
        m0 = batch['mask0_gt'][i].detach().cpu().numpy()  # ★
        m1 = batch['mask1_gt'][i].detach().cpu().numpy()  # ★
        # else:
        #     # if pred mask
        #     m0 = pred_mask0_bin[i].detach().cpu().numpy()
        #     m1 = pred_mask1_bin[i].detach().cpu().numpy()

        if len(mkpts0) == 0:
            print("error,len(mkpts0)<0 after loftr")
            continue

        # # 按掩码过滤关键点
        # in_mask = (m0[mkpts0[:, 1].round().astype(int),
        # mkpts0[:, 0].round().astype(int)] > 0) & \
        #           (m1[mkpts1[:, 1].round().astype(int),
        #           mkpts1[:, 0].round().astype(int)] > 0)
        # mkpts0 = mkpts0[in_mask]
        # mkpts1 = mkpts1[in_mask]
        #
        # if len(mkpts0) < 3:
        #     print("error,len(mkpts0)<3 after filtering")
        #     R_preds.append(torch.eye(3, device=device))
        #     t_preds.append(torch.zeros(1, 3, device=device))
        #     continue

        # ✅ 按掩码过滤关键点
        in_mask = (m0[mkpts0[:, 1].round().astype(int),
        mkpts0[:, 0].round().astype(int)] > 0) & \
                  (m1[mkpts1[:, 1].round().astype(int),
                  mkpts1[:, 0].round().astype(int)] > 0)

        mkpts0 = mkpts0[in_mask]
        mkpts1 = mkpts1[in_mask]
        mconf = mconf[in_mask]  # ★ 同步过滤置信度

        # -------------------------
        # 4）按置信度排序，取 top-8
        # -------------------------
        idx = np.argsort(-mconf)[:8]
        mkpts0 = mkpts0[idx]
        mkpts1 = mkpts1[idx]
        mconf = mconf[idx]

        # -------------------------
        # 5）画匹配可视化图
        # -------------------------
        img0_vis = (img0.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img1_vis = (img1.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        concat = np.concatenate([img0_vis, img1_vis], axis=1)

        for (p0, p1) in zip(mkpts0, mkpts1):
            p1_shift = p1.copy()
            p1_shift[0] += img0_vis.shape[1]  # shift x of image1
            cv = (255, 0, 0)
            cv2.line(concat, tuple(p0.astype(int)), tuple(p1_shift.astype(int)), cv, 1)

        Image.fromarray(concat).save(f"{save_dir}/match_vis_{i}.png")

    print(f"[debug_loftr] saved {num_save} samples to {save_dir}")

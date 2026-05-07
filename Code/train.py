import torch


def train(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
):

    model.train()
    total_samples = 0


    for batch in train_loader:

        if len(batch) == 6:
            _, depth, inertial, skeleton, y, subnet_flag = batch
        elif len(batch) == 5:
            depth, inertial, skeleton, y, subnet_flag = batch
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

        inertial = inertial.to(device)
        skeleton = skeleton.to(device)
        depth = depth.to(device)
        y = y.to(device)

        bsz = y.shape[0]
        total_samples += bsz

        # ---- forward / loss / backward ----
        optimizer.zero_grad(set_to_none=True)

        model = model.to(device)
        logits = model(skeleton, inertial, depth)
   
        loss = criterion(logits, y)
        # print("loss:", loss.item())
        # print("requires_grad:", y_pred.requires_grad)

        loss.backward()
        optimizer.step()

    return model.state_dict(), total_samples


@torch.no_grad()
def evaluate(
        model,
        data_loader,
        criterion,
        device,
):
    """
    Returns:
      acc, loss
    Notes:
      - Assumes batch is: (color, depth, inertial, skeleton, y, subnet_flag)
      - Uses final output: chunks_out[-1]
      - Computes mean loss over samples, and accuracy over samples
    """
    model.eval()

    total = 0
    correct = 0
    total_loss = 0.0

    for batch in data_loader:
        # ---- unpack batch ----
        if len(batch) == 6:
            _, depth, inertial, skeleton, y, subnet_flag = batch
        elif len(batch) == 5:
            depth, inertial, skeleton, y, subnet_flag = batch
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

        depth = depth.to(device, non_blocking=True)
        inertial = inertial.to(device, non_blocking=True)
        skeleton = skeleton.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # bsz = y.shape[0]
        # total_samples += bsz

        # subnet_flag: take first row (client-fixed)
        # if isinstance(subnet_flag, torch.Tensor):
        #     flag = subnet_flag[0].tolist()
        # else:
        #     flag = subnet_flag[0] if isinstance(subnet_flag[0], (list, tuple)) else subnet_flag

        # chunks_out, _ = model(
        #     depth=depth,
        #     inertial=inertial,
        #     skeleton=skeleton,
        #     subnet_flag=flag,
        #     blockAtt=blockAtt,
        #     mode="eval",
        # )

        logits = model(skeleton, inertial, depth)
        loss = criterion(logits, y)

        # y_pred = chunks_out[-1]

        # loss = criterion(y_pred, y)
        # total_loss += float(loss.item()) * bsz

        total_loss += loss.item() * y.size(0)

        _, predicted = torch.max(logits, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        # pred_label = torch.argmax(y_pred, dim=1)
        # total_correct += int((pred_label == y).sum().item())

    # avoid divide-by-zero
    # if total_samples == 0:
    #     return 0.0, 0.0

    avg_loss = total_loss / total
    acc = (correct / total) * 100.0

    # acc = (total_correct / total_samples) * 100.0
    # avg_loss = total_loss / total_samples
    return acc, avg_loss

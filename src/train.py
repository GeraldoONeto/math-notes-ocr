import time
import torch
import mlflow

def move_batch(batch, device, dtype):
    batch = {k: v.to(device=device, dtype=dtype) for k, v in batch.items()}
    return batch

def validate(model, loader, device):
    model.eval()
    total, n = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device, model.dtype)
            total += model(**batch).loss.item()
            n+=1

    model.train()
    return total/n

def save_checkpoint(path, model, optimizer, epoch):
    weights = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(
        {'model_state': weights, 'optimizer_state': optimizer.state_dict(), 'epoch': epoch},
        path
    )

def train(model, train_loader, val_loader, device, epochs, lr, accum_steps, ckpt_dir, run_name):
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    mlflow.set_experiment("got-ocr2-finetuning")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "epochs": epochs, "lr": lr, "accum_steps": accum_steps, "batch_size": train_loader.batch_size
        })
        best_val = float("inf")
        model.train()

        for epoch in range(epochs):
            running_loss, t0 = 0.0, time.time()
            for i, batch in enumerate(train_loader):
                batch = move_batch(batch, device, model.dtype)
                loss = model(**batch).loss
                (loss/accum_steps).backward()
                running_loss += loss.item()

                if (i + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if (i+1) % 100 == 0:
                    print(f"epoch {epoch} step {i+1}/{len(train_loader)} "
                          f"loss {running_loss/(i+1):.4f} ({(time.time()-t0)/(i+1):.2f} s/step)")
                if (i+1) % 500 == 0:
                    save_checkpoint(f"{ckpt_dir}/last.pt", model, optimizer, epoch)

            
            val_loss = validate(model, val_loader, device)
            mlflow.log_metrics(
                {"train_loss": running_loss / len(train_loader), "val_loss": val_loss}, step=epoch
            )
            print(f"== epoch {epoch}: val_loss {val_loss:.4f}")

            save_checkpoint(f"{ckpt_dir}/last.pt", model, optimizer, epoch)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(f"{ckpt_dir}/best.pt", model, optimizer, epoch)
                print("   best checkpoint saved")
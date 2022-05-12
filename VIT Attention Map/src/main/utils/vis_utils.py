import visdom


class VisTool:
    def __init__(self, use_visdom):
        if use_visdom:
            self.visdom = visdom.Visdom()
        else:
            self.visdom = None

    def plot_meters(self, mode, loss, top1_acc, top5_acc, mAP, confusion, step, lr=None):
        if self.visdom is not None:
            self.visdom.line(Y=[loss], X=[step], win=f"Loss/{mode}",
                update="append", opts={"title": f"{mode} loss", "legend": [f"{mode} loss"]}
            )
            self.visdom.line(Y=[[top1_acc, top5_acc]], X=[step], win=f"Acc/{mode}",
                update="append", opts={"title": f"{mode} Accuracy", "legend": ["top1-acc", "top5-acc"]}
            )
            self.visdom.line(Y=[mAP], X=[step], win=f"mAP/{mode}",
                update="append", opts={"title": f"{mode} mAP", "legend": [f"{mode} mAP"]}
            )
            self.visdom.heatmap(confusion, win=f"Confusion/{mode}", opts={"title": f"{mode} Confusion Matrix"})

            if lr is not None:
                self.visdom.line(Y=[lr], X=[step], win=f"lr",
                    update="append", opts={"title": "Learning Rate", "legend": ["lr"]}
                )

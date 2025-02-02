import torch
import torch.nn.functional as F
from torch import Tensor

from logreg import LogReg


def masked_accuracy(logits: Tensor, labels: Tensor):
    if len(logits) == 0:
        return 0
    pred = torch.argmax(logits, dim=1)
    acc = pred.eq(labels).sum() / len(logits) * 100
    return acc.item()

def accuracy(logits, labels):
    if len(logits) == 0:
        return 0
    pred = torch.argmax(logits, dim=1)
    acc = pred.eq(labels).sum() / len(logits) * 100
    
    acc = acc.item()
    
    return acc


def linear_evaluation(z, labels, masks, lr=0.01, max_epoch=100, mode='test'):
    z = z.detach()
    hid_dim, num_classes = z.shape[1], int(labels.max()) + 1

    classifier = LogReg(hid_dim, num_classes).to(z.device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.0)

    for epoch in range(1, max_epoch + 1):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)

        logits = classifier(z[masks[0]])
        loss = F.cross_entropy(logits, labels[masks[0]])
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        classifier.eval()
        logits = classifier(z)
        if mode == 'valid':
            accs = accuracy(logits[masks[1]], labels[masks[1]])
        elif mode == 'test':
            accs = accuracy(logits[masks[2]], labels[masks[2]])
        else:
            print('not defined error')

    return accs


def linear_evaluation_other(z, labels, masks, lr=0.01, max_epoch=100, dataset='cora', mode='test'):
    z = z.detach()
    hid_dim, num_classes = z.shape[1], int(labels.max()) + 1

    classifier = LogReg(hid_dim, num_classes).to(z.device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-9)

    patience, val_acc, wait = 10, 0, 0

    for epoch in range(1, max_epoch + 1):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)

        logits = classifier(z[masks[0]])
        loss = F.cross_entropy(logits, labels[masks[0]])
        if epoch % 10 == 0:
            classifier.eval()
            val_logits = classifier(z[masks[1]])
            dev = accuracy(val_logits, labels[masks[1]])
            if dev > val_acc:
                val_acc = dev
                wait = 0
                torch.save(classifier.state_dict(), './savepoint/'+dataset+'_classifier.pkl')
            else:
                wait += 1
            if wait > patience:
                #print("Early Stopping!")
                break

        loss.backward()
        optimizer.step()

    classifier.load_state_dict(torch.load('./savepoint/'+dataset+'_classifier.pkl'))
    classifier.eval()
    logits = classifier(z)
    if mode == 'valid':
        accs = accuracy(logits[masks[1]], labels[masks[1]])
    elif mode == 'test':
        accs = accuracy(logits[masks[2]], labels[masks[2]])
    else:
        print('not defined error')

    return accs
from config import FLAGS
from model import Model
from utils_our import get_flag
from utils import Timer
from torch.utils.data import DataLoader
from batch import BatchData
import torch


def train(train_data, saver):
    print('creating model...')
    model = Model(train_data)
    model = model.to(FLAGS.device)
    print(model)
    saver.log_model_architecture(model)
    model.train_data = train_data
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    num_epochs = get_flag('num_epochs')
    num_epochs = num_epochs if num_epochs is not None else int(1e5)
    num_iters_total = 0
    for epoch in range(num_epochs):
        loss, num_iters_total, stop = _train_epoch(
            epoch, num_iters_total, train_data, model, optimizer, saver)
        if stop:
            break
    return model


def _train_epoch(epoch, num_iters_total, data, model, optimizer, saver):
    epoch_timer = Timer()
    iter_timer = Timer()
    data_loader = DataLoader(data, batch_size=FLAGS.batch_size, shuffle=True)
    total_loss = 0
    num_iters = 0
    stop = False
    for iter, batch_gids in enumerate(data_loader):
        batch_data = BatchData(batch_gids, data.dataset)
        loss = _train_iter(
            iter, num_iters_total, epoch, batch_data, data_loader, model,
            optimizer, saver, iter_timer)
        # print(iter, batch_data, batch_data.num_graphs, len(loader.dataset))
        total_loss += loss
        num_iters += 1
        num_iters_total += 1
        if num_iters == FLAGS.only_iters_for_debug:
            stop = True
            break
        num_iters_total_limit = get_flag('num_iters')
        if num_iters_total_limit is not None and \
                num_iters_total == num_iters_total_limit:
            stop = True
            break
    saver.log_tvt_info('Epoch: {:03d}, Loss: {:.7f} ({} iters)\t\t{}'.format(
        epoch + 1, total_loss / num_iters, num_iters,
        epoch_timer.time_msec_and_clear()))
    return total_loss, num_iters_total, stop


def _train_iter(iter, num_iters_total, epoch, batch_data, data_loader, model,
                optimizer, saver, iter_timer):
    model.train()
    model.zero_grad()
    loss = model(batch_data)
    loss.backward()
    optimizer.step()
    saver.writer.add_scalar('loss/loss', loss, epoch * len(data_loader) + iter)
    loss = loss.item()
    if iter == 0 or (iter + 1) % FLAGS.print_every_iters == 0:
        saver.log_tvt_info('\tIter: {:03d} ({}), Loss: {:.7f}\t\t{}'.format(
            iter + 1, num_iters_total + 1, loss,
            iter_timer.time_msec_and_clear()))
    return loss


def test(data, model, saver):
    model.eval()
    model.test_data = data
    _test_loop(data, model, saver)


def _test_loop(data, model, saver):
    data_loader = DataLoader(data, batch_size=FLAGS.batch_size, shuffle=False)
    num_iters = 0
    iter_timer = Timer()
    for iter, batch_gids in enumerate(data_loader):
        batch_data = BatchData(batch_gids, data.dataset)
        loss = model(batch_data)
        loss = loss.item()
        saver.log_tvt_info('\tIter: {:03d}, Test Loss: {:.7f}\t\t{}'.format(
            iter + 1, loss, iter_timer.time_msec_and_clear()))
        num_iters += 1
        if num_iters == FLAGS.only_iters_for_debug:
            break

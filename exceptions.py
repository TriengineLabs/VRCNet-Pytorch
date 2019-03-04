class StopTrainingException(Exception):
    def __init__(self, ep_num):
        mess = 'Training is interrupted by user at epoch %d' % ep_num
        super().__init__(mess)

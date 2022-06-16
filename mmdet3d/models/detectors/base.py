# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.core import Det3DDataSample
from mmdet3d.core.utils import (ForwardResults, InstanceList, OptConfigType,
                                OptMultiConfig, OptSampleList, SampleList)
from mmdet3d.registry import MODELS
from mmdet.models import BaseDetector


@MODELS.register_module()
class Base3DDetector(BaseDetector):
    """Base class for 3D detectors.

    Args:
       data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
       init_cfg (dict or ConfigDict, optional): the config to control the
           initialization. Defaults to None.
    """

    def __init__(self,
                 data_processor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(data_preprocessor=data_processor, init_cfg=init_cfg)

    def forward(self,
                batch_inputs_dict: dict,
                batch_data_samples: OptSampleList = None,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            batch_inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(batch_inputs_dict, batch_data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(batch_inputs_dict, batch_data_samples,
                                **kwargs)
        elif mode == 'tensor':
            return self._forward(batch_inputs_dict, batch_data_samples,
                                 **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def convert_to_datasample(self, results_list: InstanceList) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            results_list (list[:obj:`InstanceData`]): Detection results of
                each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of 3D bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C >=7.
        """
        out_results_list = []
        for i in range(len(results_list)):
            result = Det3DDataSample()
            result.pred_instances_3d = results_list[i]
            out_results_list.append(result)
        return out_results_list

from PartSeg_smfish.copy_labels import CopyLabelWidget


class TestCopyLabels:
    def test_create(self, make_napari_viewer, qtbot):
        widget = CopyLabelWidget(make_napari_viewer())
        qtbot.addWidget(widget)

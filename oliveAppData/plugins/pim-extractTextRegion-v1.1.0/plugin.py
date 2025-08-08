"""
Plugin/Pimiento used to do conditional processing  of LID scores into a ASR plugin/domain(s)
"""

import json
import operator
from olive.plugins import *
from distutils.util import strtobool


PLUGIN_MAPPING = "mapping"

import importlib
SCORE_THRESHOLD_KEY = 'score_threshold'

OPTION_DEFAULTS = {SCORE_THRESHOLD_KEY : 0.0}

class CustomPlugin(Plugin, DataOutputTransformer):

    def __init__(self):
        # Task name may need some improvement
        self.task = "PIM"
        self.label = "Pimiento to convert Region score class IDs into text data input for MT plugin"
        self.description = "Pimiento used to create text input from ASR region scores."
        self.vendor = "SRI"
        self.version = '1.1.0'
        self.minimum_runtime_version = "5.3.0"
        self.minimum_olive_version = "5.3.0"
        self.create_date = "2021-8-06"
        self.revision_date = "2021-8-06"
        self.group = "Olive"

        # values unique to this plugin:
        self.loaded_domains = []

        # This plugin is for internal use only, so it should NOT be advertised to clients
        self.protected = True
        self.count = 0

    def list_classes(self, domain_id):
        # do we have classes that we support?
        return []

    def load(self, domain_id):
        # Doesn't do much...  just parses the domain specific mapping files
        self.count = 0
        if domain_id not in self.loaded_domains:
            # Define domains
            domain = self.get_domains()[domain_id]
            # load the class id to plugin/domain mappings for for this plugin
            domain.map_config = dict()
            self._parse_mapping_config(domain)
            self.loaded_domains.append(domain_id)


    def run_data_transformation(self, domain_id, scores, workspace, classes=None, opts=None):

        # Needs to take in regions scores and return a (list?) string value

        # set or receive a score threshold... also need an option to pick the top score, top N scores, or all scores?
        threshold = OPTION_DEFAULTS[SCORE_THRESHOLD_KEY]

        # Check for any options:
        if opts is not None:
            threshold = float(opts[SCORE_THRESHOLD_KEY]) if SCORE_THRESHOLD_KEY in opts else threshold

        # Don't think we need a domain for this...
        domain = self.get_domains()[domain_id]

        # scores = region_scores
        # something like:
        # test_scores = {
        #     0: [(0.1, 2.5, 'hello', 1.0)],
        #     1: [(3.8, 6.5, 'asr', 0.6)],
        #     2: [(6.5, 10.1, 'world', 1.1)]
        # }

        # Verify we received the ASR region scores
        logger.debug("Received region scores: {}".format(scores))

        # Check if it's a dictioanry from a workflow result
        try:
            scores = scores.get_result()
        except:
            pass

        text_list = []
        for rs_index in scores:
            region_score = scores[rs_index]
            for start_t, end_t, class_id, score in region_score:
                if score > threshold:
                    if text_list:
                        if domain_id == 'asr-mt-sent-v1':
                            text_list.append("\n")
                        else:
                            text_list.append(" ")
                    text_list.append(class_id)

        # Return the string... this is a node that produces data...
        # return ''.join(text_list)
        return {'data': ''.join(text_list)}  # No options returned

    def get_data_output_transformer_opts(self):
        """
        These options are used in the OLIVE GUI and may be configured on the commandline by passing a file to --options
        """

        trait_options = [
            TraitOption(SCORE_THRESHOLD_KEY, "Score Threshold",
                        "A Global scores must meet or exceed this threshold to select a plugin.",
                        TraitType.CHOICE_TRAIT, "", OPTION_DEFAULTS[SCORE_THRESHOLD_KEY]),
        ]
        return trait_options

    def _parse_mapping_config(self, domain):

        domain.map_config[PLUGIN_MAPPING] = dict()
        with open(os.path.join(domain.get_path(), 'mapping.json'), 'r', encoding="utf8") as f:
            raw_config = json.load(f)
            # print("plugin loaded config:", raw_config)

            for class_label in raw_config:
                # print("found:", class_label)
                if len(raw_config[class_label]) == 2:
                    domain.map_config[PLUGIN_MAPPING][class_label] = (raw_config[class_label][0], raw_config[class_label][1])
                else:
                    self.escape_with_error("Error parsing domain ({}) mapping configuration.  Expected a plugin and "
                                           "domain name but found '{}' for class: '{}'"
                                           .format(domain.get_id(), raw_config[class_label], class_label))

plugin = CustomPlugin()




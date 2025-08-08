"""
Plugin/Pimiento used to do conditional processing  of LID scores into a ASR plugin/domain(s)
"""

import ast
import json
import operator
from olive.plugins import *
from distutils.util import strtobool


minimum_runtime_version = "5.2.0"
PLUGIN_MAPPING = "mapping"

import importlib

# todo may
SCORE_THRESHOLD_KEY = 'region_threshold'
SELECT_BEST_KEY = 'select_best'
SDD_SCORES = 'sdd_scores'

OPTION_DEFAULTS = {SCORE_THRESHOLD_KEY : 0.0, SELECT_BEST_KEY: True}

class CustomPlugin(Plugin, PluginToPlugin):

    def __init__(self):
        # Task name may need some improvement
        self.task = "P2P "
        # self.label = "ASR Plugin selection using LDD (Region) Scores and SDD (Region) Scores? "
        #self.label = "Plugin selection using Region Scores "
        self.label = "MAP Conditional Workflow Router for Region Scorers"
        self.description = "Pimento used to select a plugin/domain based on region scores."
        self.vendor = "SRI"
        self.version = '1.1.1'
        self.minimum_runtime_version = "5.2.0"
        self.minimum_olive_version = "5.2.0"
        self.create_date = "2021-8-4"
        self.revision_date = "2021-8-4"
        self.group = "Olive"

        # values unique to this plugin:
        self.loaded_domains = []

        # This plugin is for internal use only, so it should NOT be advertised to clients
        self.protected = True


    def list_classes(self, domain_id):
        # should not have classes to support...
        return []

    def load(self, domain_id):
        # Doesn't do much...  just parses the domain specific mapping files
        if domain_id not in self.loaded_domains:
            # Define domains
            domain = self.get_domains()[domain_id]
            # load the class id to plugin/domain mappings for for this plugin
            domain.map_config = dict()
            self._parse_mapping_config(domain)
            self.loaded_domains.append(domain_id)


    def run_plugin_analysis(self, domain_id, region_scores, workspace, classes=None, opts=None):
        # set or receive a score threshold... also need an option to pick the top score, top N scores, or all scores?
        threshold = OPTION_DEFAULTS[SCORE_THRESHOLD_KEY]
        top_pick_only = OPTION_DEFAULTS[SELECT_BEST_KEY]

        # Check for any options:
        if opts is not None:
            threshold = float(opts[SCORE_THRESHOLD_KEY]) if SCORE_THRESHOLD_KEY in opts else threshold
            top_pick_only = bool(strtobool(opts[SELECT_BEST_KEY]) )if SELECT_BEST_KEY in opts \
                else top_pick_only

            sdd_scores = opts[SDD_SCORES] if SDD_SCORES in opts else None

            logger.info("CLG asr select best: {}".format(top_pick_only))
            if sdd_scores:
                logger.info("CLG sdd regions scores: {}".format(sdd_scores))
            else:
                logger.warn("NO SDD scores")
                # is this an error?

        logger.info("CLG asr chooser options:, {}={}, {}={}".format(SCORE_THRESHOLD_KEY, threshold, SELECT_BEST_KEY, top_pick_only))

        domain = self.get_domains()[domain_id]
        if 'mapping' in opts:
            mapping_overrides = ast.literal_eval(opts['mapping'])
            for override in mapping_overrides:
                domain.map_config[PLUGIN_MAPPING][list(override.keys())[0]] = override[list(override.keys())[0]]

        # The domain should have a lookup table like:
        # lookup_table = {'eng': ('lid-asr-test', 'english'),
        #                 # 'eng': ('lid-asr-dynapy', 'eng-tdnnChain-tel-v1'),
        #                 'cmn': ('lid-asr-dynapy', 'cmn-tdnnChain-tel-v1'),
        #                 'fas': ('lid-asr-dynapy', 'fas-tdnn-tel-v1'),
        #                 'rus': ('lid-asr-dynapy', 'rus-tdnnChain-tel-v1'),
        #                 'spa': ('lid-asr-dynapy', 'spa-tdnnChain-tel-v1')}

        logger.info("region_scores: {}".format(region_scores))
        language_regions = {} # list()
        for rs in region_scores:
            for start_t, end_t, class_id, score in region_scores[rs]:
                if score > threshold:
                    if class_id in domain.map_config[PLUGIN_MAPPING]:
                        pd = domain.map_config[PLUGIN_MAPPING][class_id]
                        logger.info(language_regions)
                        if pd[1] not in language_regions.keys():
                            language_regions[pd[1]] = {'plugin': pd[0], 'domain':pd[1], 'region':[(start_t, end_t)]}
                        else:
                            language_regions[pd[1]]['region'].append((start_t, end_t))
                    else:
                        info_msg = "Unable to choose a plugin/domain since '{}' is not one of the supported class names: {}".format(class_id,
                                                                      list(domain.map_config[PLUGIN_MAPPING].keys()))
                        logger.info(info_msg)

        plugin_domains = list()
        for field in language_regions.keys():
            plugin_domains.append(language_regions[field])

        logger.debug("Final selected plugin domains: {}".format(plugin_domains))

        # validate there is a  score
        if len(plugin_domains) == 0:
            self.escape_with_error("Score values too low to select a plugin/domain.")

        return plugin_domains


    def get_plugin_selection_opts(self):
        """
        These options are used in the OLIVE GUI and may be configured on the commandline by passing a file to --options
        """

        trait_options = [
            TraitOption(SCORE_THRESHOLD_KEY, "Score Threshold",
                        "A Global scores must meet or exceed this threshold to select a plugin.",
                        TraitType.CHOICE_TRAIT, "", OPTION_DEFAULTS[SCORE_THRESHOLD_KEY]),
            TraitOption(SELECT_BEST_KEY,
                        "Pick Top Score", "Select the best plugin/domain for the highest scoring result, otherwise "
                                          "return a selection for all scores that meet the score threshold",
                        TraitType.CHOICE_TRAIT, "", OPTION_DEFAULTS[SELECT_BEST_KEY]),
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




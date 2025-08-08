import os
import json
from chooser import plugin as plugin_root
# from chooser import plugin as plugin_root


# import olive.server.oliveserver as oliveserver
import olive.core.env as env
# from olive.server.workflow.order import WorkflowOrderWrapper
# from olive.server.workflow.order import OrderState
# import olive.server.messaging.olive_pb2 as olive_proto
# from olive.plugins.trait import WorkflowResult

import olive.plugins.plugin as plug
from unittest import TestCase
from unittest import SkipTest

def hello():
    return "Midge"

class TestPlugin(TestCase):

    def setUp(self):
        # makes sure OLIVE_APP_DATA is set and it contains a plugin
        if env.get_app_data_path is None:
            self.skipTest("Environment variable 'OLIVE_APP_DATA' must be set to run these tests, skipping test....")

        plugin_path = env.get_plugins_path()
        if not os.path.exists(plugin_path):
            self.skipTest("Plugin path '{}' is not valid, skipping test....".format(plugin_path))


    def test_asr_lid_mapping(self):
        # open config - make sure we get the expected values
        # with open('../domains/lid-asr/mapping.json', 'r', encoding="utf8") as f:
        #     config = json.load(f)
        #     print(config)

        # we assume these unit tests are ran from $OLIVE/test/app_data/plugins...

        plugin_id = 'chooser'
        domain_id = 'test-lid-asr'
        # load the plugin/domain in the standard way:
        plug._load_domain(plugin_id, domain_id)
        # now get the plugin
        plugin = plug._plugins_by_id[plugin_id]

        # get domain
        domain = plugin.get_domains()[domain_id]
        plugin_domain_map = domain.config[plugin_root.PLUGIN_MAPPING]
        self.assertEqual(len(plugin_domain_map), 5)
        english_pd = plugin_domain_map['eng']
        self.assertEqual(english_pd[0],  'lid-asr-test')
        self.assertEqual(english_pd[1],  'english')

        spanish_pd = plugin_domain_map['spa']
        self.assertEqual(spanish_pd[0],  'lid-asr-dynapy')
        self.assertEqual(spanish_pd[1],  'spa-tdnnChain-tel-v1')

        # classes
        class_ids = plugin.list_classes(domain_id)
        self.assertEqual(len(class_ids), 5)
        keys = ['eng', 'cmn', 'fas', 'rus', 'spa']
        for k in keys:
            self.assertTrue(k in class_ids)



        # plugin.parse_mapping_config(domain_id)


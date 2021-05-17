import xlrd
import openpyxl
from pathlib import Path
# noinspection PyUnresolvedReferences
from xml.dom import minidom

trace_link_file = 'artifactWrappers.xmi'
trace_model_file = 'traceModel.xmi'
hazard_file = 'mobstr-hazards.xlsx'
requirement_file = 'mobstr-requirements.xlsx'
req_file_without_preprocessing = 'resource/org.panorama-research.mobstr.requirements/mobstr-requirements.xlsx'


class Node:
    def __init__(self, id=None):
        self.id = id
        self.text = None
        self.target = []


class Failure(Node):
    pass


class Hazard(Node):
    pass


class SafetyGoal(Node):
    pass


class SafetyRequirement(Node):
    pass


def artifactWrapperHandler():
    xmldoc = minidom.parse(trace_link_file)
    sub_grpaph_link = []
    for element in xmldoc.getElementsByTagName('artifacts'):
        if 'platform' in element.getAttribute('path'):
            id = element.getAttribute('path').partition('row=')
            path = element.getAttribute('path').split('platform:/')[1].split('/mobstr-')[0]
            if 'hazard' in path:
                file = Path(path, hazard_file)
                obj = openpyxl.load_workbook(file, data_only=True)
                sheet = obj.active
                for row in range(1, sheet.max_row + 1):
                    if sheet.cell(row=row, column=1).value == id[2]:
                        string = id[2] + ' ' + sheet.cell(row=row, column=2).value
                        sub_grpaph_link.append(string)
            elif 'requirement' in path:
                file = Path(path, requirement_file)
                obj = openpyxl.load_workbook(file, data_only=True)
                sheet = obj.active
                for row in range(1, sheet.max_row + 1):
                    if sheet.cell(row=row, column=1).value == id[2]:
                        string = id[2] + ' ' + sheet.cell(row=row, column=3).value
                        sub_grpaph_link.append(string)
        else:
            sub_grpaph_link.append(element.getAttribute('path'))

    return sub_grpaph_link


def add_failure():
    failure_list = []
    xmldoc = minidom.parse(trace_model_file)
    failure_id = 0
    for element in xmldoc.getElementsByTagName('traces'):
        failure_id += 1
        failure = Failure(failure_id)
        if 'Failure' in element.getAttribute('name'):
            failure.text = element.getAttribute('name').partition(': Failure ->')[0]
            failure_list.append(failure)
            add_hazard(element, failure)
    return failure_list


def add_hazard(element, failure):
    dict_h = {}
    sub_grpaph_link = artifactWrapperHandler()
    for e1 in element.getElementsByTagName('target'):
        p1 = e1.getAttribute('href')
        id1 = p1.partition('@artifacts.')
        textInList1 = sub_grpaph_link[(int(id1[2]))]
        hazard_id = id1[2]
        if hazard_id not in dict_h:
            hazard = Hazard(hazard_id)
            hazard.text = textInList1
            current_hazard_id = hazard.text.split(' ')[0]
            dict_h[hazard_id] = hazard
        else:
            hazard = dict_h[hazard_id]
        failure.target.append(hazard)
        add_safety_goal(hazard, hazard_id, current_hazard_id)


def add_safety_goal(hazard, hazard_id, current_hazard_id):
    dict_s = {}
    safety_goal_id = hazard_id

    if safety_goal_id not in dict_s:
        file = Path(req_file_without_preprocessing)
        obj = openpyxl.load_workbook(file, data_only=True)
        sheet = obj.active
        for row in range(1, sheet.max_row + 1):
            if sheet.cell(row=row, column=2).value == current_hazard_id:
                safety_goal = SafetyGoal(safety_goal_id)
                safety_goal.text = sheet.cell(row=row, column=3).value
                dict_s[safety_goal_id] = safety_goal
                hazard.target.append(safety_goal)

    else:
        safety_goal = dict_s[safety_goal_id]
    add_safety_requirement(safety_goal, safety_goal_id, current_hazard_id)


def add_safety_requirement(safety_goal, safety_goal_id, current_hazard_id):
    dict_r = {}
    requirement_id = safety_goal_id
    if requirement_id not in dict_r:
        id_row_number = 0
        file = Path(requirement_file)
        obj = openpyxl.load_workbook(file, data_only=True)
        sheet = obj.active
        for findid in range(1, sheet.max_row):
            if sheet.cell(row=findid, column=2).value is not None and current_hazard_id in sheet.cell(
                    row=findid, column=2).value:
                id_row_number = findid
                for row in range(id_row_number + 1, sheet.max_row):
                    if sheet.cell(row=row, column=1).value is not None and 'SR' in sheet.cell(row=row,
                                                                                              column=1).value:
                        requirement = SafetyRequirement(requirement_id)
                        requirement.text = sheet.cell(row=row, column=3).value.replace('The system shall ', '')
                        dict_r[requirement_id] = requirement
                        safety_goal.target.append(requirement)
                    if sheet.cell(row=row, column=1).value is not None and 'SG' in sheet.cell(row=row,
                                                                                              column=1).value:
                        break

    else:
        requirement = dict_r[requirement_id]


def print_everything(failure_list):
    for failure in failure_list:
        print()
        print("FAILURE")
        # print("Id")
        # print(failure.id)
        print("Failure Text: " + failure.text)
        # time.sleep(0.2)
        print_hazards(failure.target)


def print_hazards(hazards):
    for hazard in hazards:
        print("HAZARD")
        # print("Id")
        # print(hazard.id)
        print("Hazard Text: " + hazard.text)
        # time.sleep(0.2)

        print_safety_goals(hazard.target)


def print_safety_goals(safety_goals):
    for safety_goal in safety_goals:
        print("SAFETY GOAL")
        # print("id")
        # print(safety_goal.id)
        print("Safety goal Text: " + safety_goal.text)
        # time.sleep(0.2)
        print_requirements(safety_goal.target)


def print_requirements(requirements):
    for requirement in requirements:
        print("REQUIREMENT")
        # print("id")
        # print(requirement.id)
        # print(requirement.id)
        print("Safety requirement Text: ")
        print(requirement.text)
        # time.sleep(0.2)


if __name__ == '__main__':
    l = add_failure()
    # print_everything(l)

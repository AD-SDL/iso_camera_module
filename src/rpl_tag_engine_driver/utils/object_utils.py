"""
Module for creating instances of Database Entries, Databases, Measurements, Detections, and others.
"""

import sqlite3

import matplotlib.pyplot as plt
import numpy as np


class DBEntry:
    """
    A base class for database entries with a name attribute.
    """

    def __init__(self, name: str = ""):
        """
        Initialize a new database entry with the given name.

        Args:
            name (str, optional): The name of the database entry. Default is an empty string.
        """
        self.name = name

    def __str__(self) -> str:
        """
        Return the string representation of the database entry.

        Returns:
            str: The name of the database entry.
        """
        return self.name


class BaseDatabase:
    """
    A base class for managing a collection of database entries with unique IDs and names.
    """

    def __init__(self):
        """
        Initialize an empty database with no entries.
        """
        self.next_ID = 0
        self.max_ID = -1
        self.entries = {}
        self.lookup = {}

    def __str__(self, verbose: bool = False) -> str:
        """
        Return a string representation of the database.

        Args:
            verbose (bool): If True, includes detailed information about each entry. Default is False.

        Returns:
            str: A string representing the database.
        """
        res = f"{self.next_ID},{len(self.entries)},{len(self.lookup)}\n"
        if verbose:
            for ID in self.entries:
                res += f"{ID},{self.entries[ID].name}\n"
        return res

    def clear(self):
        """
        Clear all entries and reset the database to its initial state.
        """
        self.next_ID = 0
        self.max_ID = -1
        self.entries.clear()
        self.lookup.clear()

    def add_by_id(self, entry_id: int, new_entry: DBEntry, entity_name: str = "entity"):
        """
        Add a new entry to the database by a specific ID.

        Args:
            entry_id (int): The ID for the new entry.
            new_entry (DBEntry): The new entry to add.
            entity_name (str): The name of the entity being added. Default is 'entity'.

        Returns:
            bool: True if the entry was added successfully, False otherwise.
        """
        if entry_id in self.entries:
            print(f"ERROR: cannot overwrite {entity_name} with ID {entry_id}.")
            return False
        if new_entry.name in self.lookup:
            print(f"ERROR: {entity_name} with name {new_entry.name} already exists.")
            return False
        self.entries[entry_id] = new_entry
        self.lookup[new_entry.name] = entry_id
        if entry_id > self.max_ID:
            self.max_ID = entry_id
            self.next_ID = self.max_ID + 1
        return True

    def add(self, new_entry: DBEntry, entity_name: str = "entity") -> bool:
        """
        Add a new entry to the database with an automatically assigned ID.

        Args:
            new_entry (DBEntry): The new entry to add.
            entity_name (str): The name of the entity being added. Default is 'entity'.

        Returns:
            bool: True if the entry was added successfully, False otherwise.
        """
        return self.add_by_id(self.next_ID, new_entry, entity_name)

    def exists(self, name: str) -> bool:
        """
        Check if an entry with the given name exists in the database.

        Args:
            name (str): The name to check for existence.

        Returns:
            bool: True if an entry with the given name exists, False otherwise.
        """
        return name in self.lookup

    def id_exists(self, entry_id: int) -> bool:
        """
        Check if an entry with the given ID exists in the database.

        Args:
            entry_id (int): The ID to check for existence.

        Returns:
            bool: True if an entry with the given ID exists, False otherwise.
        """
        return entry_id in self.entries


class MultitagTemplate(DBEntry):
    """
    A class representing a template for multitags, containing multiple tag slots.
    """

    def __init__(self, name: str = "SINGLE", offset: list = None, scale: float = 1.0, theta: float = 0.0):
        """
        Initialize a MultitagTemplate with a default tag slot.

        Args:
            name (str): The name of the template.
            offset (list): The offset of the first tag slot.
            scale (float): The scale of the first tag slot.
            theta (float): The rotation of the first tag slot in degrees.
        """
        super().__init__(name)
        self.tag_slots = []
        self.add_tag_slot(offset or [0.0, 0.0], scale, theta)

    def __str__(self) -> str:
        """
        Return a string representation of the MultitagTemplate.

        Returns:
            str: A string representing the template, including its name, number of tag slots, and details of each slot.
        """
        res = f"{self.name:10} {len(self.tag_slots):2} "
        for slot in self.tag_slots:
            res += f"[ [{slot[0][0]:5.2f}, {slot[0][1]:5.2f} ], {slot[1]:5.2f}, {slot[2]:4.0f} ] "
        return res

    def tag_corners_relative(self, offset: list, scale: float, theta: float) -> np.ndarray:
        """
        Calculate the relative positions of tag corners based on offset, scale, and rotation.

        Args:
            offset (list): The offset of the tag slot.
            scale (float): The scale of the tag slot.
            theta (float): The rotation of the tag slot in degrees.

        Returns:
            np.ndarray: An array of corner positions.
        """
        ox = offset[0] * 2 / scale
        oy = offset[1] * 2 / scale
        theta_rad = np.radians(theta)
        s = np.sin(theta_rad)
        c = np.cos(theta_rad)

        corners = (
            np.array(
                [
                    -(c + s) + ox,
                    -(c - s) - oy,
                    (c - s) + ox,
                    -(c + s) - oy,
                    (c + s) + ox,
                    (c - s) - oy,
                    -(c - s) + ox,
                    (c + s) - oy,
                ]
            ).reshape(-1, 2)
            * 0.5
            * scale
        )

        return corners

    def add_tag_slot(self, offset: list = None, scale: float = 1.0, theta: float = 0.0):
        """
        Add a new tag slot to the template.

        Args:
            offset (list): The offset of the new tag slot.
            scale (float): The scale of the new tag slot.
            theta (float): The rotation of the new tag slot in degrees.
        """
        offset = offset or [0.0, 0.0]
        corners = self.tag_corners_relative(offset, scale, theta)
        self.tag_slots.append([offset, scale, theta, corners])

    def opoints(self, tag_zero_size: float, slot_num: int = -1) -> np.ndarray:
        """
        Get the corner points of the tag(s) in 3D space.

        Args:
            tag_zero_size (float): The size of the base tag.
            slot_num (int): The slot number to get the points for. If -1, all slots are used.

        Returns:
            np.ndarray: An array of corner points in 3D space.
        """
        if slot_num < 0:
            opoints = np.vstack([slot[3] for slot in self.tag_slots])
        elif slot_num < len(self.tag_slots):
            opoints = self.tag_slots[slot_num][3]
        else:
            print(f"ERROR: slot number {slot_num} is out of range")
            opoints = self.tag_slots[0][3]

        N = opoints.shape[0]
        opoints = np.hstack((opoints * tag_zero_size, np.zeros((N, 1)))).reshape(N, 1, 3)
        return opoints


class MultitagTemplateDatabase(BaseDatabase):
    """
    A database class for managing multitag templates.
    Inherits from BaseDatabase and provides methods for adding,
    retrieving, saving, and loading multitag templates.
    """

    def __str__(self, verbose=False):
        """
        Return a string representation of the database.
        If verbose is True, includes detailed information about each template.
        """
        res = f"Next ID: {self.next_ID}, Entries: {len(self.entries)}\n"
        if verbose:
            for template_id, template in self.entries.items():
                res += f"{template_id}: {template}\n"
        return res

    def add(self, new_template):
        """
        Add a new template to the database.

        Args:
            new_template (MultitagTemplate): The template to add.

        Returns:
            int: The ID of the newly added template.
        """
        return super().add(new_template, entity_name="template")

    def add_by_signature(self, new_template_signature):
        """
        Add a new template to the database by its signature.

        Args:
            new_template_signature (tuple): A tuple containing the template signature.

        Returns:
            int: The ID of the newly added template.
        """
        name, first_anchor = new_template_signature[:2]
        new_template = MultitagTemplate(name=name, offset=first_anchor[0], scale=first_anchor[1], theta=first_anchor[2])

        for anchor in new_template_signature[2:]:
            new_template.add_tag_slot(offset=anchor[0], scale=anchor[1], theta=anchor[2])
        return self.add(new_template)

    def ID_of_named_template(self, template_name):
        """
        Get the ID of a template by its name.

        Args:
            template_name (str): The name of the template.

        Returns:
            int: The ID of the template, or -1 if not found.
        """
        template_id = self.lookup.get(template_name)
        if template_id:
            return template_id
        print(f'ERROR: Template with name "{template_name}" not found')
        return -1

    def by_name(self, template_name):
        """
        Get a template by its name.

        Args:
            template_name (str): The name of the template.

        Returns:
            MultitagTemplate: The template object, or -1 if not found.
        """
        template_id = self.lookup.get(template_name)
        if template_id:
            return self.entries[template_id]
        print(f'ERROR: Template with name "{template_name}" not found')
        return -1

    def by_ID(self, template_id):
        """
        Get a template by its ID.

        Args:
            template_id (int): The ID of the template.

        Returns:
            MultitagTemplate: The template object, or -1 if not found.
        """
        template = self.entries.get(template_id)
        if template:
            return template
        print(f"ERROR: Template with ID {template_id} not found")
        return -1

    def load(self, template_database_filename):
        """
        Load templates from a database file into memory.

        Args:
            template_database_filename (str): The filename of the database to load from.
        """
        con = sqlite3.connect(template_database_filename)
        cur = con.cursor()

        self.clear()

        query = """
        SELECT template_ID, template_name, slot_number, total_slots,
               offset_x, offset_y, scale, theta
        FROM templates
        """
        template_records = cur.execute(query).fetchall()
        for r in template_records:
            template_id, name, slot_number, total_slots, offset_x, offset_y, scale, theta = r

            if not self.id_exists(template_id):
                new_template = MultitagTemplate(name=name)
                new_template.tag_slots = [None] * total_slots
                super().add_by_id(template_id, new_template, entity_name="multitag")

            slot_value = [
                [offset_x, offset_y],
                scale,
                theta,
                new_template.tag_corners_relative(offset=[offset_x, offset_y], scale=scale, theta=theta),
            ]
            self.entries[template_id].tag_slots[slot_number] = slot_value

        con.close()

    def save(self, template_database_filename):
        """
        Save templates from memory to a database file.

        Args:
            template_database_filename (str): The filename of the database to save to.
        """
        con = sqlite3.connect(template_database_filename)
        cur = con.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS templates(
            template_ID INTEGER,
            template_name TEXT,
            slot_number INTEGER,
            total_slots INTEGER,
            offset_x REAL,
            offset_y REAL,
            scale REAL,
            theta REAL
        )
        """)
        data = []
        for template_id, template in self.entries.items():
            template_name = template.name
            total_slots = len(template.tag_slots)
            for slot_number, slot in enumerate(template.tag_slots):
                data.append(
                    (template_id, template_name, slot_number, total_slots, slot[0][0], slot[0][1], slot[1], slot[2])
                )

        cur.executemany("INSERT INTO templates VALUES (?, ?, ?, ?, ?, ?, ?, ?)", data)

        con.commit()
        con.close()


class Multitag(DBEntry):
    """
    A class representing a multitag, which is an entry that contains a list of tag IDs
    and is associated with a specific template.
    """

    def __init__(self, name=None):
        """
        Initialize a Multitag instance.

        Args:
            name (str, optional): The name of the multitag. Defaults to None.
        """
        super(Multitag, self).__init__(name)
        self.tags = []
        self.multitag_template = None

    def __str__(self):
        """
        Return a string representation of the multitag.

        Returns:
            str: The string representation of the multitag.
        """
        res = f"{self.name:20} {self.multitag_template:10} [ "
        res += " ".join(f"{int(ID):5}" for ID in self.tags)
        return res + "]"

    def is_tag_used(self, tag_ID):
        """
        Check if a specific tag ID is used in the multitag.

        Args:
            tag_ID (int): The tag ID to check.

        Returns:
            bool: True if the tag ID is used, False otherwise.
        """
        return tag_ID in self.tags


class MultitagDatabase(BaseDatabase):
    """A database class for handling multitag entities with support for membership tracking."""

    def __init__(self):
        """Initialize the MultitagDatabase with an empty membership dictionary."""
        super().__init__()
        self.membership = {}

    def __str__(self, verbose=False):
        """
        Return a string representation of the database.

        Args:
            verbose (bool): If True, include details of each entry.

        Returns:
            str: The string representation of the database.
        """
        res = f"next_ID {self.next_ID}, entries {len(self.entries)}, membership entries {len(self.membership)}\n"
        if verbose:
            for ID in self.entries:
                res += f"{ID:3}  {self.entries[ID]}\n"
        return res

    def clear(self):
        """Clear the database and membership dictionary."""
        super().clear()
        self.membership.clear()

    def update_membership(self, multitag_ID, multitag: Multitag):
        """
        Update the membership dictionary with the multitag.

        Args:
            multitag_ID (int): The ID of the multitag.
            multitag (Multitag): The multitag instance to update.
        """
        for tag_ID in multitag.tags:
            if tag_ID not in self.membership:
                self.membership[tag_ID] = []
            self.membership[tag_ID].append(multitag_ID)

    def add(self, multitag: Multitag):
        """
        Add a multitag to the database and update membership.

        Args:
            multitag (Multitag): The multitag instance to add.

        Returns:
            bool: True if the multitag was added successfully, False otherwise.
        """
        multitag_ID = self.next_ID
        success_flag = super().add(multitag, entity_name="multitag")
        if success_flag:
            self.update_membership(multitag_ID, multitag)
        return success_flag

    def add_by_name(self, multitag_name, tag_ID_list, template_name):
        """
        Add a multitag to the database by specifying its name, tags, and template.

        Args:
            multitag_name (str): The name of the multitag.
            tag_ID_list (list[int]): The list of tag IDs associated with the multitag.
            template_name (str): The name of the template associated with the multitag.

        Returns:
            bool: True if the multitag was added successfully, False otherwise.
        """
        new_multitag = Multitag()
        new_multitag.name = multitag_name
        new_multitag.tags = tag_ID_list
        new_multitag.multitag_template = template_name
        return self.add(new_multitag)

    def load(self, multitag_database_filename):
        """
        Load multitags from a database file.

        Args:
            multitag_database_filename (str): The filename of the database to load.
        """
        con = sqlite3.connect(multitag_database_filename)
        cur = con.cursor()

        self.clear()

        res = cur.execute("SELECT multitag_ID, multitag_name, tag_ID_list, template_name FROM multitags")
        multitag_records = res.fetchall()
        for r in multitag_records:
            new_multitag = Multitag()
            multitag_ID = r[0]
            new_multitag.name = r[1]
            new_multitag.tags = [int(tid) for tid in r[2].split(",")]
            new_multitag.multitag_template = r[3]
            super().add_by_id(multitag_ID, new_multitag, entity_name="multitag")
            self.update_membership(multitag_ID, new_multitag)

        con.close()

    def save(self, multitag_database_filename):
        """
        Save all multitags to a database file.

        Args:
            multitag_database_filename (str): The filename of the database to save to.
        """
        con = sqlite3.connect(multitag_database_filename)
        cur = con.cursor()

        cur.execute("CREATE TABLE multitags(multitag_ID, multitag_name, tag_ID_list, template_name)")

        data = []
        for multitag_ID in self.entries:
            multitag_name = self.entries[multitag_ID].name
            tag_ID_list = ",".join([str(tID) for tID in self.entries[multitag_ID].tags])
            template_name = self.entries[multitag_ID].multitag_template
            data.append((multitag_ID, multitag_name, tag_ID_list, template_name))

        cur.executemany("INSERT INTO multitags VALUES(?,?,?,?)", data)

        con.commit()
        con.close()

    def is_tag_used(self, tag_ID):
        """
        Check if a tag is used in any multitag.

        Args:
            tag_ID (int): The tag ID to check.

        Returns:
            bool: True if the tag is used, False otherwise.
        """
        for mt in self.entries:
            if self.entries[mt].is_tag_used(tag_ID):
                return True
        return False

    def match_multitag(self, tag_ID_list):
        """
        Match a multitag based on the provided list of tag IDs.

        Args:
            tag_ID_list (list[int]): The list of tag IDs to match.

        Returns:
            str: The name of the matching multitag, or an empty string if no match is found.
        """
        if not tag_ID_list:
            return ""

        for mt in self.entries:
            if self.entries[mt].is_tag_used(tag_ID_list[0]):
                if len(tag_ID_list) == len(self.entries[mt].tags):
                    if all(tid in self.entries[mt].tags for tid in tag_ID_list):
                        return self.entries[mt].name
        return ""


class LabObject(DBEntry):
    """
    A class for representing a lab object with associated multitag IDs and a frame.
    """

    def __init__(self, name: str = ""):
        """
        Initialize a new LabObject instance.

        Args:
            name (str): The name of the lab object. Default is an empty string.
        """
        super().__init__(name)
        self.frame = None
        self.multitag_IDs = []

    def __str__(self, verbose: bool = False) -> str:
        """
        Return a string representation of the lab object.

        Args:
            verbose (bool): If True, includes detailed information about the lab object. Default is False.

        Returns:
            str: A string representation of the lab object.
        """
        res = f"LabObject(name={self.name}, frame={self.frame}, multitag_IDs={self.multitag_IDs})"
        if verbose:
            res += f"\nMultitag IDs: {self.multitag_IDs}"
        return res

    def add_multitag(self, multitag_ID: int):
        """
        Add a multitag ID to the list of associated multitag IDs.

        Args:
            multitag_ID (int): The ID of the multitag to be added.
        """
        self.multitag_IDs.append(multitag_ID)

    def set_frame(self, multitag_ID: int) -> bool:
        """
        Set the frame of the lab object to the specified multitag ID.

        Args:
            multitag_ID (int): The ID of the multitag to set as the frame.

        Returns:
            bool: True if the frame was set successfully, False otherwise.
        """
        if multitag_ID not in self.multitag_IDs:
            print(f"ERROR: multitag with ID {multitag_ID} is not associated with this LabObject")
            return False
        self.frame = multitag_ID
        return True

    def is_tag_used(self, tag_ID: int) -> bool:
        """
        TBD
        """
        # TBD
        pass


class ObjectDatabase(BaseDatabase):
    """
    A class for managing a database of objects, inheriting from BaseDatabase.
    """

    def __str__(self):
        """
        TBD
        """
        # TBD
        pass

    def clear(self):
        """
        Clear all entries in the object database.
        """
        super().clear()

    def add(self, new_obj):
        """
        Add a new object to the database.

        Args:
            new_obj: The object to be added.

        Returns:
            int: The ID assigned to the new object.
        """
        return super().add(new_obj, entity_name="object")

    def load(self, database_filename):
        """
        TBD
        """
        # TBD
        pass

    def save(self, database_filename):
        """
        TBD
        """
        # TBD
        pass


class camera_model(DBEntry):
    """
    A class representing a camera model with calibration parameters.

    Attributes:
        camera_matrix (np.ndarray): The camera matrix for intrinsic parameters.
        distortion_params (np.ndarray): The distortion coefficients for lens distortion.
    """

    def __init__(self, camera_matrix, distortion_params):
        """
        Initialize a camera model with the given camera matrix and distortion parameters.

        Args:
            camera_matrix (np.ndarray): The camera matrix for intrinsic parameters.
            distortion_params (np.ndarray): The distortion coefficients for lens distortion.
        """
        self.camera_matrix = camera_matrix
        self.distortion_params = distortion_params
        return


class CameraDatabase(BaseDatabase):
    """
    TBD
    """

    def __str__(self, verbose=False):
        """
        TBD
        """
        # TBD
        return ""

    def add(self, new_camera):
        """
        TBD
        """
        return super().add(new_camera, entity_name="camera")

    def load(self, database_filename):
        """
        TBD
        """
        # TBD
        pass

    def save(self, database_filename):
        """
        TBD
        """
        # TBD
        pass


class Measurement(DBEntry):
    """
    A class for representing a measurement associated with a reference and target multitag.
    """

    def __init__(
        self,
        name: str = "",
        datestamp: str = "",
        reference_multitag_ID: int = None,
        target_multitag_ID: int = None,
        image_file: str = "",
        camera_file: str = "",
        relative_rvec: np.ndarray = None,
        relative_tvec: np.ndarray = None,
        reference_record: list = None,
        target_record: list = None,
    ):
        """
        Initialize a new Measurement instance.

        Args:
            name (str): The name of the measurement. Default is an empty string.
            datestamp (str): The date and time when the measurement was taken. Default is an empty string.
            reference_multitag_ID (int): The ID of the reference multitag. Default is None.
            target_multitag_ID (int): The ID of the target multitag. Default is None.
            image_file (str): The filename of the image associated with the measurement. Default is an empty string.
            camera_file (str): The filename of the camera configuration associated with the measurement. Default is an empty string.
            relative_rvec (np.ndarray): The relative rotation vector. Default is None.
            relative_tvec (np.ndarray): The relative translation vector. Default is None.
            reference_record (list): The reference record containing additional data. Default is None.
            target_record (list): The target record containing additional data. Default is None.
        """
        super().__init__(name)
        self.datestamp = datestamp
        self.reference_multitag_ID = reference_multitag_ID
        self.target_multitag_ID = target_multitag_ID
        self.image_file = image_file
        self.camera_file = camera_file
        self.relative_rvec = relative_rvec if relative_rvec is not None else np.zeros(3)
        self.relative_tvec = relative_tvec if relative_tvec is not None else np.zeros(3)
        self.reference_record = reference_record if reference_record is not None else []
        self.target_record = target_record if target_record is not None else []

    def __str__(self) -> str:
        """
        Return a string representation of the measurement.

        Returns:
            str: A string representing the measurement details.
        """
        return (
            f"Measurement(name={self.name}, datestamp={self.datestamp}, "
            f"reference_multitag_ID={self.reference_multitag_ID}, "
            f"target_multitag_ID={self.target_multitag_ID}, "
            f"image_file={self.image_file}, camera_file={self.camera_file}, "
            f"relative_rvec={self.relative_rvec}, relative_tvec={self.relative_tvec}, "
            f"reference_record={self.reference_record}, target_record={self.target_record})"
        )


class MeasurementDatabase:
    """
    A class for managing a database of measurements.
    """

    def __init__(self):
        """
        Initialize an empty measurement database.
        """
        self.next_ID = 0
        self.max_ID = -1
        self.entries = {}
        self.lookup = {}

    def __str__(self, verbose=False):
        """
        TBD
        """
        # TBD
        return ""

    def clear(self):
        """
        Clear all entries and reset IDs in the measurement database.
        """
        self.next_ID = 0
        self.max_ID = -1
        self.entries.clear()
        self.lookup.clear()

    def add(self, multitag_ref, multitag_other, new_meas):
        """
        Add a new measurement to the database.

        Args:
            multitag_ref (int): The reference ID of the first multitag.
            multitag_other (int): The reference ID of the second multitag.
            new_meas (list): A list containing measurement details.

        Returns:
            bool: True if the measurement was added successfully, False otherwise.
        """
        if self.next_ID in self.lookup:
            return False

        self.lookup[self.next_ID] = [multitag_ref, multitag_other]

        if multitag_ref not in self.entries:
            self.entries[multitag_ref] = {}

        if multitag_other not in self.entries[multitag_ref]:
            self.entries[multitag_ref][multitag_other] = []

        self.entries[multitag_ref][multitag_other].append([self.next_ID] + new_meas)
        self.max_ID = self.next_ID
        self.next_ID += 1

        return True

    def load(self, measurement_database_filename):
        """
        Load measurements from a database file into the measurement database.

        Args:
            measurement_database_filename (str): The filename of the database to load.
        """
        con = sqlite3.connect(measurement_database_filename)
        cur = con.cursor()

        self.clear()

        res = cur.execute("""
            SELECT measurement_ID, multitag_ref, multitag_targ, img_fname, cam_fname, rvx, rvy, rvz, tvx, tvy, tvz
            FROM measurements
        """)

        measurement_records = res.fetchall()

        for r in measurement_records:
            measurement_ID = r[0]
            multitag_ref = r[1]
            multitag_other = r[2]
            img_fname = r[3]
            cam_fname = r[4]
            r_vec = np.array([r[5], r[6], r[7]])
            t_vec = np.array([r[8], r[9], r[10]])

            measurement_record = [measurement_ID, img_fname, cam_fname, r_vec, t_vec]

            if multitag_ref not in self.entries:
                self.entries[multitag_ref] = {}

            if multitag_other not in self.entries[multitag_ref]:
                self.entries[multitag_ref][multitag_other] = []

            self.entries[multitag_ref][multitag_other].append(measurement_record)
            self.lookup[measurement_ID] = [multitag_ref, multitag_other]

            if measurement_ID > self.max_ID:
                self.max_ID = measurement_ID
                self.next_ID = self.max_ID + 1

        con.close()

    def save(self, measurement_database_filename):
        """
        Save all measurements from the measurement database to a file.

        Args:
            measurement_database_filename (str): The filename to save the database to.
        """
        con = sqlite3.connect(measurement_database_filename)
        cur = con.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                measurement_ID INTEGER PRIMARY KEY,
                multitag_ref INTEGER,
                multitag_targ INTEGER,
                img_fname TEXT,
                cam_fname TEXT,
                rvx REAL,
                rvy REAL,
                rvz REAL,
                tvx REAL,
                tvy REAL,
                tvz REAL
            )
        """)

        data = []
        for multitag_ref in self.entries:
            for multitag_other in self.entries[multitag_ref]:
                for record in self.entries[multitag_ref][multitag_other]:
                    measurement_ID = record[0]
                    img_fname = record[1]
                    cam_fname = record[2]
                    rvx, rvy, rvz = record[3]
                    tvx, tvy, tvz = record[4]

                    data.append(
                        (
                            measurement_ID,
                            int(multitag_ref),
                            int(multitag_other),
                            img_fname,
                            cam_fname,
                            rvx,
                            rvy,
                            rvz,
                            tvx,
                            tvy,
                            tvz,
                        )
                    )

        cur.executemany(
            """
            INSERT INTO measurements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            data,
        )

        con.commit()
        con.close()


class RPLTagDatabase:
    """
    A class for managing databases of multitag templates, multitags, objects, and measurements.
    """

    def __init__(self):
        """
        Initialize the RPLTagDatabase instance with empty databases for templates, multitags, objects, and measurements.
        """
        self.templates = MultitagTemplateDatabase()
        self.multitags = MultitagDatabase()
        self.objects = ObjectDatabase()
        self.measurements = MeasurementDatabase()

    def __str__(self, verbose: bool = False) -> str:
        """
        Return a string representation of the RPLTagDatabase.

        Args:
            verbose (bool): If True, includes detailed information about each database component. Default is False.

        Returns:
            str: A string representing the RPLTagDatabase.
        """
        res = (
            self.templates.__str__(verbose=verbose)
            + "\n"
            + self.multitags.__str__(verbose=verbose)
            + "\n"
            + self.objects.__str__(verbose=verbose)
            + "\n"
            + self.measurements.__str__(verbose=verbose)
        )
        return res

    def new_template(self, template_signature: str) -> bool:
        """
        Add a new template to the template database.

        Args:
            template_signature (str): The signature of the new template.

        Returns:
            bool: True if the template was added successfully, False otherwise.
        """
        return self.templates.add_by_signature(template_signature)

    def new_multitag(self, multitag_name: str, tag_ID_list: list, template_name: str) -> bool:
        """
        Add a new multitag to the multitag database.

        Args:
            multitag_name (str): The name of the new multitag.
            tag_ID_list (list): A list of tag IDs associated with the multitag.
            template_name (str): The name of the template associated with the multitag.

        Returns:
            bool: True if the multitag was added successfully, False otherwise.
        """
        if not self.templates.exists(template_name):
            print(f"ERROR: template with name {template_name} not found.")
            return False
        return self.multitags.add_by_name(multitag_name, tag_ID_list, template_name)

    def new_object(self, object_name: str, multitag_list: list) -> bool:
        """
        Add a new object to the object database.

        Args:
            object_name (str): The name of the new object.
            multitag_list (list): A list of multitag IDs associated with the object.

        Returns:
            bool: True if the object was added successfully, False otherwise.
        """

        # TBD, method "add_by_name" does not exist for an object.

        # return self.objects.add_by_name(object_name, multitag_list)
        pass

    def new_measurement(self, ref_multitag: int, other_multitag: int, new_meas: list) -> bool:
        """
        Add a new measurement to the measurement database.

        Args:
            ref_multitag (int): The ID of the reference multitag.
            other_multitag (int): The ID of the target multitag.
            new_meas (list): A list containing measurement data.

        Returns:
            bool: True if the measurement was added successfully, False otherwise.
        """
        return self.measurements.add(ref_multitag, other_multitag, new_meas)

    def load(self, database_filename: str):
        """
        Load all data from the specified database file.

        Args:
            database_filename (str): The path to the database file.
        """
        self.templates.load(database_filename)
        self.multitags.load(database_filename)
        self.objects.load(database_filename)
        self.measurements.load(database_filename)

    def save(self, database_filename: str):
        """
        Save all data to the specified database file.

        Args:
            database_filename (str): The path to the database file.
        """
        self.templates.save(database_filename)
        self.multitags.save(database_filename)
        self.objects.save(database_filename)
        self.measurements.save(database_filename)


def ax_plot_multitag(ax, mt: MultitagTemplate, fontsizes=[24, 18]):
    """
    Plot a multitag template on a given axis.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to plot the multitag.
        mt (MultitagTemplate): The multitag template to plot.
        fontsizes (list): A list of two font sizes for the plot title and labels. Default is [24, 18].
    """
    fs1, fs2 = fontsizes
    for slotnum, corners in enumerate(mt.tag_slots):
        ax.plot(corners[:, 0], corners[:, 1], "-")
        ax.plot(corners[0, 0], corners[0, 1], "ko")
        ax.text(np.mean(corners[:, 0]), np.mean(corners[:, 1]), str(slotnum), fontsize=fs2)

    ax.set_aspect("equal", "box")
    ax.invert_yaxis()
    ax.set_title(f"Multitag Template {mt.name}", fontsize=fs1)
    ax.set_xlabel("X coordinate", fontsize=fs2)
    ax.set_ylabel("Y coordinate (image sense)", fontsize=fs2)


def plot_multitag_template(mt: MultitagTemplate):
    """
    Plot a multitag template.

    Args:
        mt (MultitagTemplate): The multitag template to plot.
    """
    sz = 4
    fs1 = 3 * sz
    fs2 = 2.5 * sz
    plt.figure(figsize=(17, sz))
    ax_plot_multitag(plt.gca(), mt, fontsizes=[fs1, fs2])

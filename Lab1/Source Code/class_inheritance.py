"""
Part 3:
Classes here consist of Flight, Person, Passenger, Employee, Pilot, Stewardess
Examples of overriding methods and multiple level inheritance present.
A run-through of the features of the class are printed to the output.
"""


class Flight:
    registry = []
    passengers = []

    def __init__(self, flight_number, destination, departure_date, time_of_day):
        self.flight_number = flight_number
        self.destination = destination
        self.departure_date = departure_date
        self.time_of_day = time_of_day
        self.registry.append(self)

        # assign pilot and stewardess crew to this flight
        self.crew = []
        self.pilot = None
        for employee in Employee.registry:
            if type(employee) == Pilot and employee.assignment is None and self.pilot is None:
                self.pilot = employee
                employee.assignment = flight_number
            elif len(self.crew) < 3 and employee.assignment is None:
                self.crew.append(employee)
                employee.assignment = flight_number

    def printInfo(self):
        print("Flight", self.flight_number,
              "going to", self.destination,
              "on", self.departure_date,
              "at", self.time_of_day)
        print("Flown by: ", self.pilot.first_name, self.pilot.last_name)
        print("Cabin Crew: ", end="")
        for c in self.crew[:-1]:
            print(c.first_name, c.last_name, end=", ")
        print(self.crew[-1].first_name, self.crew[-1].last_name)


class Person:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

    def printName(self):
        print(self.first_name, self.last_name)

    def printInfo(self):
        self.printName()


class Passenger(Person):
    booked_flights = []

    def printFlightsAvailableOn(self, date):
        for flight in Flight.registry:
            if flight.departure_date == date:
                print("Flight " + str(flight.flight_number) +
                      " to " + flight.destination +
                      " departing at " + flight.time_of_day)

    def printFlightsTo(self, destination):
        for flight in Flight.registry:
            if flight.destination == destination:
                print("Flight " + str(flight.flight_number) +
                      " departing on the " + flight.departure_date +
                      " " + flight.time_of_day)

    def bookFlight(self, flight_num):
        for flight in Flight.registry:
            if flight.flight_number == flight_num:
                flight.passengers.append(self)
                self.booked_flights.append(flight)

    def printInfo(self):
        print(self.first_name, self.last_name, "- Passenger", end=" ")
        if len(self.booked_flights) != 0:
            print("- Has booked flight", end=" ")
            for flight in self.booked_flights:
                print(flight.flight_number, end=", ")


class Employee(Person):
    registry = []

    def __init__(self, first_name, last_name):
        super().__init__(first_name, last_name)
        self.employee_id = len(Employee.registry) + 1
        self.assignment = None
        self.registry.append(self)

    def printID(self):
        print(self.employee_id)


class Pilot(Employee):
    def printInfo(self):
        print(self.first_name, self.last_name, "- Pilot", end=" ")
        if self.assignment is not None:
            print("- Assigned to flight", self.assignment)


class Stewardess(Employee):
    def printInfo(self):
        print(self.first_name, self.last_name, "- Stewardess", end=" ")
        if self.assignment is not None:
            print("- Assigned to flight", self.assignment)


s1 = Stewardess("Sarah", "Michael")
s2 = Stewardess("Ben", "Bambi")
s3 = Stewardess("Kylie", "Menner")
s4 = Stewardess("Riley", "Dunn")
s5 = Stewardess("Can", "Man")
s6 = Stewardess("Sarah", "Rain")
s7 = Stewardess("Sofia", "Traxler")
s8 = Stewardess("Mason", "Gray")
s9 = Stewardess("Carl", "Reid")
s10 = Stewardess("John", "Hamm")

p1 = Pilot("Bob", "Kale")
p2 = Pilot("Johnny", "Wins")
p3 = Pilot("Risky", "Lazer")
p4 = Pilot("Stiny", "Poo")

f1 = Flight(1543, "Los Angles, CA", "3/23/2020", "6:00am")
f2 = Flight(8594, "Detroit, IL", "3/23/2020", "3:00pm")
f3 = Flight(7309, "Los Angles, CA", "4/2/2020", "11:00am")

pass1 = Passenger("Mike", "Wasowski")
pass2 = Passenger("Sully", "Steven")
pass3 = Passenger("Allen", "Poe")
pass4 = Passenger("Kaspi", "Cardi")


print("\tStewardess and Pilot objects:")
print("  They both inherit from the Employee class and this class gives them a unique employee ID")
print("The pilot:", end=" ")
p3.printID()
print("The stewardess:", end=" ")
s7.printID()
print()
print("  And the Employee class inherits from the Person class which gives all them name")
p3.printName()
s7.printName()
print()
print("  Both have a method overridden from their parent parent class (Person) to print their info ")
p3.printInfo()
s7.printInfo()
print()
print("\tFlight object:")
print("  When a Flight object is made it assigns a pilot and 3 stewardesses to itself if they are available")
print("  A Flight object does not have a parent class but still has a printInfo() to show all attributes:")
f2.printInfo()
print()

print("\tPassenger object:")
print("  From the Passenger object, they can check availablity of flights by date")
pass1.printFlightsAvailableOn("3/23/2020")
print("  and by destination")
pass3.printFlightsTo("Detroit, IL")
print()
print("  Passenger inherits from the Person object where its defines its name and overrides its printInfo() method")
pass2.printInfo()
print()
print("  Passenger can book flights by number which updates attributes for itself and the flight they are booking")
pass2.bookFlight(7309)
print()
print("  Now printInfo() will have updated information")
pass2.printInfo()

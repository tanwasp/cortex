//
//  Item.swift
//  Cortex
//
//  Created by Tanish Pradhan Wong Ah Sui on 10/11/25.
//
//  This file defines a data model called Item using SwiftData.
//  SwiftData is Apple's framework for storing and managing data in apps (like a database).
//  An Item represents a simple record with a timestamp, possibly for logging or history.
//  It's not actively used in the main app logic but is part of the project structure.
//

import Foundation  // Imports Foundation, which provides basic types like Date.
import SwiftData  // Imports SwiftData for data persistence.

@Model  // This attribute tells SwiftData to treat this class as a data model (it will create a database table for it).
final class Item {  // Defines a class (reference type) named Item. 'final' means it can't be subclassed.
    var timestamp: Date  // A property (variable) of type Date, storing when the item was created.
    
    init(timestamp: Date) {  // Initializer: a special function called when creating an Item. It sets the timestamp.
        self.timestamp = timestamp  // 'self' refers to the current instance; assigns the passed timestamp to the property.
    }
}
